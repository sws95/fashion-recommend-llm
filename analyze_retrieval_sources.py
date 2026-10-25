import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import os
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# Retrieval Source별 기여도·중복·커버리지 분석
# - 유저: test 기간에 새 상품(train에서 안 산 것)을 산 유저 N명, 정답 = 그 구매 1개 (evaluate_hybrid_twotower.py future와 같은 방식)
# - 소스 (각 top-K, 이미 산 상품 제외):
#   Query     : 쿼리 임베딩 (정답 속성 쿼리 → Qwen 텍스트 임베딩 전수 검색)
#   BM25      : 보강 코퍼스(상품명+종류+패턴+색상+설명) 전체 상품 대상
#   TwoTower  : Two-Tower+Qwen 개인화 (쿼리 무관)
#   BPR       : cornac BPR (app과 동일, 쿼리 무관)
#   Popular   : train 마지막 1주 판매량
# - 지표: 소스별 정답 포함률, 고유 기여(그 소스만 찾은 정답), 소스 간 후보 중복, 소스 추가 시 합집합 포함률, 신상품/기존 상품 구간

N_USERS = int(os.environ.get('N_USERS', 5000))
K = int(os.environ.get('K', 50))
device = 'cuda'

print("데이터 로딩...")
tr = pd.read_csv('./transactions_train.csv', dtype={'customer_id': str, 'article_id': str}, usecols=['t_dat', 'customer_id', 'article_id'])
tr['t_dat'] = pd.to_datetime(tr['t_dat'])
split = tr['t_dat'].quantile(0.8)
train_df, test_df = tr[tr['t_dat'] < split], tr[tr['t_dat'] >= split]
del tr
train_user_items = train_df[['customer_id', 'article_id']].drop_duplicates().groupby('customer_id')['article_id'].apply(set).to_dict()
popularity = train_df['article_id'].value_counts()
last_week = train_df[train_df['t_dat'] >= train_df['t_dat'].max() - pd.Timedelta(days=7)]['article_id'].value_counts()

item_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
ipos = {a: i for i, a in enumerate(item_ids)}
n_items = len(item_ids)
item_vec = torch.tensor(np.load('./qwen_text_emb.npy'), device=device)

# ── 개인화 소스 벡터 (전체 상품 순서로 정렬) ──
with open('./twotower_qwen_model.pkl', 'rb') as f:
    d = pickle.load(f)
assert d['item_ids'] == item_ids
tt_U, tt_umap, tt_I = torch.tensor(d['user_emb'], device=device), {u: i for i, u in enumerate(d['user_ids'])}, torch.tensor(d['item_emb'], device=device)
with open('./bpr_model.pkl', 'rb') as f:
    b = pickle.load(f)
m, ds = b['model'], b['dataset']
del b
bpr_U = torch.tensor(np.hstack([m.u_factors, np.ones((m.u_factors.shape[0], 1))]), dtype=torch.float32, device=device)
bpr_I = torch.full((n_items, bpr_U.shape[1]), 0.0, device=device)
bpr_has = torch.zeros(n_items, dtype=torch.bool, device=device)
rows = torch.tensor([ipos[a] for a in ds.iid_map], device=device)
bpr_I[rows] = torch.tensor(np.hstack([m.i_factors, m.i_biases.reshape(-1, 1)]), dtype=torch.float32, device=device)[torch.tensor(list(ds.iid_map.values()), device=device)]
bpr_has[rows] = True
pop_score = torch.zeros(n_items, device=device)
pop_score[torch.tensor([ipos[a] for a in last_week.index], device=device)] = torch.tensor(last_week.values, dtype=torch.float32, device=device)

# ── 유저 & 정답 ──
valid = set(tt_umap) & set(ds.uid_map)
tn = test_df[test_df['customer_id'].isin(valid)]
tn = tn[[a not in train_user_items[u] for u, a in zip(tn['customer_id'], tn['article_id'])]]
new_items = tn.groupby('customer_id')['article_id'].apply(lambda s: sorted(set(s)))
rng = np.random.RandomState(7)
users = rng.choice(new_items.index.values, min(N_USERS, len(new_items)), replace=False).tolist()
target = {u: rng.choice(new_items[u]) for u in users}
is_cold = {u: target[u] not in popularity.index for u in users}
print(f"유저 {len(users)}명 | 정답 중 신상품 {np.mean(list(is_cold.values())):.3f}")

# ── 쿼리 ──
art = pd.read_csv('./articles.csv', dtype={'article_id': str}).set_index('article_id').loc[item_ids]
attr = lambda a: f"{art.loc[a, 'product_type_name']} {art.loc[a, 'graphical_appearance_name']} {art.loc[a, 'colour_group_name']}"
queries = [attr(target[u]) for u in users]
model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16)
qv = []
with torch.no_grad():
    for s in tqdm(range(0, len(queries), 64), desc="쿼리 임베딩", mininterval=10):
        qv.append(model.process([{"text": t, "instruction": "Retrieve fashion items relevant to the query."} for t in queries[s:s + 64]]))
qv = torch.cat(qv).to(device).half()
del model; torch.cuda.empty_cache()

print("BM25 인덱싱 (보강 코퍼스)...")
bm25 = BM25Okapi([f"{r.prod_name} {r.product_type_name} {r.graphical_appearance_name} {r.colour_group_name} {r.detail_desc}".lower().split()
                  for r in art.itertuples()])

# ── 소스별 후보 ──
SOURCES = ['Query', 'BM25', 'TwoTower', 'BPR', 'Popular']
cands = {s: [] for s in SOURCES}
bm25_cache = {}
for i, u in enumerate(tqdm(users, desc="후보 생성", mininterval=10)):
    excl = torch.tensor([ipos[a] for a in train_user_items[u] if a in ipos], dtype=torch.long, device=device)
    def topk(score):
        score = score.float().clone()
        score[excl] = -1e9
        return set(score.topk(K).indices.tolist())
    cands['Query'].append(topk(item_vec @ qv[i]))
    if queries[i] not in bm25_cache:                       # 같은 속성 쿼리 재사용
        bm25_cache[queries[i]] = torch.tensor(bm25.get_scores(queries[i].lower().split()), device=device)
    cands['BM25'].append(topk(bm25_cache[queries[i]]))
    cands['TwoTower'].append(topk(tt_I @ tt_U[tt_umap[u]]))
    cands['BPR'].append(topk(torch.where(bpr_has, bpr_I @ bpr_U[ds.uid_map[u]], torch.tensor(-1e9, device=device))))
    cands['Popular'].append(topk(pop_score))

# ── 분석 ──
t_idx = [ipos[target[u]] for u in users]
hit = pd.DataFrame({s: [t in c for t, c in zip(t_idx, cands[s])] for s in SOURCES})
hit['cold'] = [is_cold[u] for u in users]

print(f"\n=== 소스별 정답 포함률 (top-{K}, 유저 {len(users)}명) ===")
summary = pd.DataFrame({
    '전체': hit[SOURCES].mean(),
    '신상품 정답': hit.loc[hit.cold, SOURCES].mean(),
    '기존 상품 정답': hit.loc[~hit.cold, SOURCES].mean(),
    '고유 기여 (이 소스만 찾음)': pd.Series({s: (hit[s] & ~hit[[o for o in SOURCES if o != s]].any(axis=1)).mean() for s in SOURCES}),
})
print(summary.round(4).to_string())

print(f"\n=== 소스 간 후보 중복 (평균 |A∩B| / {K}) ===")
ov = pd.DataFrame(index=SOURCES, columns=SOURCES, dtype=float)
for a in SOURCES:
    for c in SOURCES:
        ov.loc[a, c] = np.mean([len(x & y) / K for x, y in zip(cands[a], cands[c])])
print(ov.round(3).to_string())

print("\n=== 소스를 하나씩 추가할 때 합집합 포함률 (매번 가장 많이 늘리는 소스 선택) ===")
chosen, remaining = [], list(SOURCES)
while remaining:
    best = max(remaining, key=lambda s: hit[chosen + [s]].any(axis=1).mean())
    chosen.append(best); remaining.remove(best)
    sizes = np.mean([len(set().union(*[cands[s][i] for s in chosen])) for i in range(len(users))])
    print(f"  + {best:<9} 포함률 {hit[chosen].any(axis=1).mean():.4f}  "
          f"(신상품 {hit.loc[hit.cold, chosen].any(axis=1).mean():.4f} / 기존 {hit.loc[~hit.cold, chosen].any(axis=1).mean():.4f})  평균 후보 수 {sizes:.0f}")

summary.to_csv('./retrieval_sources_summary.csv')
ov.to_csv('./retrieval_sources_overlap.csv')
