import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import os
import time
import itertools
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# 추천 파이프라인(app_chain.py get_candidates) 오프라인 평가
# - 검색: Qwen 임베딩 top50 / 재정렬: (임베딩, CF, BM25) 점수 결합 / CF: BPR, TwoTower, TwoTower+Qwen
# - 프로토콜 2종
#   leak  : evaluate.py와 동일. train 구매 1개를 target (CF가 학습 때 본 아이템) → README 수치와 비교용
#   future: test 기간에 새로 산 아이템을 target (CF가 못 본 미래 구매, 신상품 포함) → 실제 성능에 가까움
#   둘 다 target 속성(타입/패턴/색상)으로 쿼리 생성
# - 가중치는 tune 유저에서 grid search → 별도 test 유저에서 보고
# - app의 epsilon 탐색(4·5등 교체) 효과, 다양성, 단계별 latency
# 차이: 아이템 인덱스는 ChromaDB(이미지+텍스트) 대신 qwen_text_emb.npy(텍스트 전용) 전수 검색

# ── 데이터 로딩 ───────────────────────────────────────────
print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str, 'article_id': str
}, usecols=['t_dat', 'customer_id', 'article_id'])
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].quantile(0.8)

train_df          = transactions[transactions['t_dat'] < split_date]
test_df           = transactions[transactions['t_dat'] >= split_date]
train_interaction = train_df[['customer_id', 'article_id']].drop_duplicates()
train_user_items  = train_interaction.groupby('customer_id')['article_id'].apply(set).to_dict()
item_popularity   = train_interaction['article_id'].value_counts().to_dict()
del transactions, train_df

articles_df  = pd.read_csv('./articles.csv', dtype={'article_id': str})
article_meta = articles_df.set_index('article_id')[
    ['product_type_name', 'graphical_appearance_name', 'colour_group_name']
].to_dict('index')

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
item_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
item_pos = {a: i for i, a in enumerate(item_ids)}
item_vec = torch.tensor(np.load('./qwen_text_emb.npy'), device=device).float()
pop_arr  = np.array([item_popularity.get(a, 1) for a in item_ids], dtype=float)   # app과 동일: 없으면 1
is_cold  = np.array([a not in item_popularity for a in item_ids])

# ── CF 모델 로딩: 각각 (유저 벡터, 유저 id→행, 전체 아이템 순서 아이템 벡터, CF에 있는 아이템 마스크) ──
def load_cf(U, uid2row, I, row2item):
    idx   = torch.tensor([item_pos[a] for a in row2item])
    I_all = torch.zeros(len(item_ids), I.shape[1])
    has   = torch.zeros(len(item_ids), dtype=torch.bool)       # CF에 없는 아이템은 점수 0 (evaluate.py와 동일)
    I_all[idx] = torch.tensor(I)
    has[idx]   = True
    return torch.tensor(U), uid2row, I_all.to(device), has.to(device)

cf_models = {}
with open('./bpr_torch_model.pkl', 'rb') as f:
    d = pickle.load(f)
cf_models['BPR'] = load_cf(d['user_emb'], d['uid_map'], d['item_emb'], [d['idx2item'][i] for i in range(len(d['idx2item']))])
with open('./twotower_model.pkl', 'rb') as f:
    d = pickle.load(f)
cf_models['TwoTower'] = load_cf(d['user_emb'], d['uid_map'], d['item_emb'], [d['idx2item'][i] for i in range(len(d['idx2item']))])
with open('./twotower_qwen_model.pkl', 'rb') as f:
    d = pickle.load(f)
cf_models['TwoTower+Qwen'] = load_cf(d['user_emb'], {u: i for i, u in enumerate(d['user_ids'])}, d['item_emb'], d['item_ids'])
# 앱에서 쓰는 cornac BPR (make_bpr_model.py, 100 iter). score = i_bias + u·i → [u, 1]·[i, b] 로 표현
if os.path.exists('./bpr_model.pkl'):
    with open('./bpr_model.pkl', 'rb') as f:
        d = pickle.load(f)
    m, ds = d['model'], d['dataset']
    U = np.hstack([m.u_factors, np.ones((m.u_factors.shape[0], 1))]).astype(np.float32)
    I = np.hstack([m.i_factors, m.i_biases.reshape(-1, 1)]).astype(np.float32)
    idx2item_c = {v: k for k, v in ds.iid_map.items()}
    cf_models['BPR (cornac, 앱)'] = load_cf(U, dict(ds.uid_map), I, [idx2item_c[i] for i in range(len(idx2item_c))])
del d

# ── 유저 & target ────────────────────────────────────────
N_USERS = int(os.environ.get('N_USERS', 2000))    # 프로토콜마다 절반 tune / 절반 test
valid_users = [
    uid for uid, items in train_user_items.items()
    if len(items) >= 2 and all(uid in m[1] for m in cf_models.values())
]

# leak: evaluate.py와 동일한 방식
np.random.seed(42)
leak_users = np.random.choice(valid_users, min(N_USERS, len(valid_users)), replace=False)
np.random.seed(42)
leak_targets = {uid: np.random.choice(list(train_user_items[uid])) for uid in leak_users}
leak_users = [u for u in leak_users if leak_targets[u] in article_meta]

# future: test 기간에 처음 산 아이템 (이미 산 아이템은 app에서 제외되므로 재구매는 target 제외)
valid_set = set(valid_users)
test_new = test_df[test_df['customer_id'].isin(valid_set) & test_df['article_id'].isin(item_pos)]
test_new = test_new[[a not in train_user_items[u] for u, a in zip(test_new['customer_id'], test_new['article_id'])]]
test_new_items = test_new.groupby('customer_id')['article_id'].apply(lambda s: sorted(set(s)))
rng = np.random.RandomState(42)
future_users   = rng.choice(test_new_items.index.values, min(N_USERS, len(test_new_items)), replace=False).tolist()
future_targets = {u: rng.choice(test_new_items[u]) for u in future_users}
del test_df, test_new

protocols = {
    'leak':   (leak_users,   leak_targets,   lambda u, t: train_user_items[u] - {t}),
    'future': (future_users, future_targets, lambda u, t: train_user_items[u]),
}
# 옵션: PROTOCOLS=future → 일부만 / DUMP_TARGETS=경로 → 정답 상품 목록만 저장하고 종료 (gen_nl_queries.py 입력)
protocols = {p: v for p, v in protocols.items() if p in os.environ.get('PROTOCOLS', 'leak,future').split(',')}
if os.environ.get('DUMP_TARGETS'):
    import json
    targets = sorted({ts[u] for us, ts, _ in protocols.values() for u in us})
    json.dump(targets, open(os.environ['DUMP_TARGETS'], 'w'))
    print(f"정답 상품 {len(targets)}개 저장: {os.environ['DUMP_TARGETS']}")
    sys.exit(0)
for p, (us, ts, _) in protocols.items():
    print(f"[{p}] 유저 {len(us)}명 | target 중 신상품 비율 {np.mean([ts[u] not in item_popularity for u in us]):.3f}")

# ── 쿼리 임베딩 ──────────────────────────────────────────
print("Embedding 모델 로딩...")
embedding_model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16)
INSTRUCTION = "Retrieve fashion items relevant to the query."

# 옵션: QUERY_FILE=nl_queries.json → 정답 속성 쿼리 대신 LLM이 만든 자연어 요청 사용 (gen_nl_queries.py)
nl_queries = {}
if os.environ.get('QUERY_FILE'):
    import json
    nl_queries = json.load(open(os.environ['QUERY_FILE'], encoding='utf-8'))
    print(f"자연어 쿼리 {len(nl_queries)}개 사용: {os.environ['QUERY_FILE']}")

def make_query(aid):
    if aid in nl_queries:
        return nl_queries[aid]
    m = article_meta[aid]
    return f"{m['product_type_name']} {m['graphical_appearance_name']} {m['colour_group_name']}"

all_queries = sorted({make_query(ts[u]) for us, ts, _ in protocols.values() for u in us})
q_index = {q: i for i, q in enumerate(all_queries)}
q_vec = []
with torch.no_grad():
    for s in tqdm(range(0, len(all_queries), 64), desc="쿼리 임베딩"):
        q_vec.append(embedding_model.process([{"text": t, "instruction": INSTRUCTION} for t in all_queries[s:s + 64]]).float())
q_vec = torch.cat(q_vec).to(device)

# ── BM25 (make_bm25_model.py와 같은 코퍼스) ──────────────
print("BM25 인덱싱...")
bm25 = BM25Okapi([(str(r.prod_name) + ' ' + str(r.detail_desc)).lower().split()
                  for r in articles_df.set_index('article_id').loc[item_ids].itertuples()])

# ── 단계별 latency (app 기준, 유저 1명 요청) ─────────────
def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000

sample_q = all_queries[0]
cand50   = (item_vec @ q_vec[0]).topk(50).indices.tolist()
U0, m0, I0, h0 = cf_models['TwoTower+Qwen']
latency = {
    '쿼리 임베딩 (Qwen, 1건)':            timeit(lambda: embedding_model.process([{"text": sample_q, "instruction": INSTRUCTION}]), 5),
    '임베딩 전수검색 top50 (GPU)':         timeit(lambda: (item_vec @ q_vec[0]).topk(50)),
    'BM25 get_scores 전체 (app 현재)':     timeit(lambda: bm25.get_scores(sample_q.lower().split()), 5),
    'BM25 get_batch_scores 후보 50개':     timeit(lambda: bm25.get_batch_scores(sample_q.lower().split(), cand50)),
    'CF 전체 아이템 스코어링 (GPU)':        timeit(lambda: I0 @ U0[0].to(device)),
}
del embedding_model
torch.cuda.empty_cache()

# ── 공통 함수 ────────────────────────────────────────────
N_RETRIEVE = 50
SIZES      = (50, 100, 200, 500, 1000)            # 후보 수 실험 (쿼리 임베딩 top-N)
K_LIST     = (5, 10)
EPSILON    = 0.1                                  # app_chain.py
POP_ALPHA  = 0.01

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def rrf(arr, k=60):
    ranks = np.empty(len(arr)); ranks[np.argsort(-arr)] = np.arange(len(arr))
    return 1 / (k + ranks + 1)

def list_metrics(ranked, tpos):
    out = {}
    for k in K_LIST:
        top = list(ranked[:k])
        if tpos >= 0 and tpos in top:
            r = top.index(tpos)
            out[k] = (1.0, 1 / np.log2(r + 2), 1 / (r + 1))
        else:
            out[k] = (0.0, 0.0, 0.0)
    return out

def apply_explore(ranked, scores, e, c, cand_pop, eps, rng):
    """app_chain.py와 동일: 확률 eps로 4등 ← 인기 패널티 1등, 5등 ← emb 높고 CF 낮은 아이템"""
    ranked = list(ranked)
    if rng.random() < eps:
        p = int(np.argmax(scores / (cand_pop ** POP_ALPHA)))
        if p not in ranked[:3]:
            ranked[3] = p
    if rng.random() < eps:
        x = int(np.argmax(e - c))
        if x not in ranked[:4]:
            ranked[4] = x
    return ranked

GRID = [(we / 10, wc / 10, round(1 - we / 10 - wc / 10, 1))
        for we, wc in itertools.product(range(11), repeat=2) if we + wc <= 10]
FIXED = {'Embedding only': (1, 0, 0), 'BM25 only': (0, 0, 1),
         '기존 hybrid (.3/.7/0)': (0.3, 0.7, 0), '기존 app (.5/.3/.2)': (0.5, 0.3, 0.2)}

rows, div_rows, seg_rows, size_rows = [], [], [], []
for proto, (p_users, p_targets, remaining_fn) in protocols.items():
    # 후보 & 임베딩/BM25 점수는 CF 모델과 무관 → 유저별 한 번만
    users = []
    for uid in tqdm(p_users, desc=f"[{proto}] 후보/BM25"):
        target    = p_targets[uid]
        remaining = {item_pos[a] for a in remaining_fn(uid, target) if a in item_pos}
        q         = make_query(target)
        emb_scores = item_vec @ q_vec[q_index[q]]
        cand_big = [i for i in emb_scores.topk(max(SIZES) + len(remaining)).indices.tolist() if i not in remaining][:max(SIZES)]
        cand = cand_big[:N_RETRIEVE]
        tpos = cand.index(item_pos[target]) if item_pos[target] in cand else -1
        n_hist = len(train_user_items[uid])
        users.append({'uid': uid, 'target': item_pos[target], 'remaining': remaining, 'cand': np.array(cand), 'tpos': tpos,
                      'e': emb_scores[cand].cpu().numpy(), 'qi': q_index[q],
                      'cand_big': np.array(cand_big), 'e_big': emb_scores[cand_big].cpu().numpy(),
                      'tpos_big': cand_big.index(item_pos[target]) if item_pos[target] in cand_big else -1,
                      'bm': np.array(bm25.get_batch_scores(q.lower().split(), cand)),
                      'cold': target not in item_popularity,
                      'hist': '1-5' if n_hist <= 5 else '6-20' if n_hist <= 20 else '21+'})
    n_tune = len(users) // 2
    tune, test = range(n_tune), range(n_tune, len(users))
    recall50 = np.mean([u['tpos'] >= 0 for u in users[n_tune:]])

    for name, (U, uid2row, I_all, has) in cf_models.items():
        feats, cf_only, c_big = [], [], []
        for u in tqdm(users, desc=f"[{proto}] {name}"):
            cf_scores = torch.where(has, I_all @ U[uid2row[u['uid']]].to(device), torch.zeros((), device=device))
            cf_full = cf_scores.clone()
            if u['remaining']:
                cf_full[torch.tensor(list(u['remaining']), device=device)] = -1e9
            top = cf_full.topk(N_RETRIEVE).indices.tolist()
            cf_only.append(list_metrics(top, top.index(u['target']) if u['target'] in top else -1))
            c = cf_scores[torch.tensor(u['cand'], device=device)].cpu().numpy()
            c_big.append(cf_scores[torch.tensor(u['cand_big'], device=device)].cpu().numpy())
            # 리트리벌 실험: 쿼리 임베딩 top50 ∪ CF 개인화 top50 → 같은 방식으로 재정렬
            union = np.array(list(dict.fromkeys(u['cand'].tolist() + top)))
            union_t = torch.tensor(union, device=device)
            e_u = (item_vec[union_t] @ q_vec[u['qi']]).cpu().numpy()
            c_u = cf_scores[union_t].cpu().numpy()
            tpos_u = int(np.where(union == u['target'])[0][0]) if u['target'] in union else -1
            feats.append({'minmax': (normalize(u['e']), normalize(c), normalize(u['bm'])),
                          'rrf':    (rrf(u['e']), rrf(c), rrf(u['bm'])), 'tpos': u['tpos'], 'cand': u['cand'],
                          'union': (normalize(e_u), normalize(c_u), np.zeros(len(union))),
                          'tpos_union': tpos_u, 'cand_union': union})

        def rank_lists(idx_range, w, fusion, eps=0.0, seed=0):
            rng = np.random.RandomState(seed)
            for i in idx_range:
                f = feats[i]
                s = sum(wi * x for wi, x in zip(w, f[fusion]))
                ranked = np.argsort(-s, kind='stable')
                if eps > 0:
                    e, c, _ = f['minmax']
                    ranked = apply_explore(ranked, s, e, c, pop_arr[f['cand']], eps, rng)
                yield i, ranked

        def evaluate(idx_range, w, fusion, **kw):
            out = {k: [] for k in K_LIST}
            tkey = 'tpos_union' if fusion == 'union' else 'tpos'
            for i, ranked in rank_lists(idx_range, w, fusion, **kw):
                m = list_metrics(ranked, feats[i][tkey])
                for k in K_LIST:
                    out[k].append(m[k])
            return {k: np.mean(v, axis=0) for k, v in out.items()}

        def per_user(idx_range, w, fusion):
            tkey = 'tpos_union' if fusion == 'union' else 'tpos'
            return [(i, list_metrics(r, feats[i][tkey])[10]) for i, r in rank_lists(idx_range, w, fusion)]

        def diversity(idx_range, w, fusion, **kw):
            ckey = 'cand_union' if fusion == 'union' else 'cand'
            tops = [feats[i][ckey][list(r[:10])] for i, r in rank_lists(idx_range, w, fusion, **kw)]
            flat = np.concatenate(tops)
            return {'coverage@10': len(set(flat)) / len(flat), 'avg_log_pop@10': np.log1p(pop_arr[flat]).mean(),
                    'cold_share@10': is_cold[flat].mean()}

        def report(mode, res, w=None, div=None):
            for k in K_LIST:
                rows.append({'protocol': proto, 'CF': name, 'mode': mode, 'weights(e/cf/bm)': w, 'k': k,
                             **dict(zip(['HitRate', 'NDCG', 'MRR'], res[k]))})
            if div:
                div_rows.append({'protocol': proto, 'CF': name, 'mode': mode, **div})

        fmt = lambda w: '/'.join(f'{x:g}' for x in w)
        report('CF only', {k: np.mean([cf_only[i][k] for i in test], axis=0) for k in K_LIST})
        for mode, w in FIXED.items():
            report(mode, evaluate(test, w, 'minmax'), fmt(w), diversity(test, w, 'minmax'))
        best_overall = None
        for fusion in ('minmax', 'rrf'):
            for label, grid in (('emb+cf', [g for g in GRID if g[2] == 0]), ('emb+cf+bm25', GRID)):
                tune_scores = {w: evaluate(tune, w, fusion)[10][1] for w in grid}        # tune NDCG@10 기준
                best = max(tune_scores, key=tune_scores.get)
                report(f'tuned {label} [{fusion}]', evaluate(test, best, fusion), fmt(best), diversity(test, best, fusion))
                if best_overall is None or tune_scores[best] > best_overall[0]:
                    best_overall = (tune_scores[best], best, fusion)
        # 리트리벌 실험: 후보 = 쿼리 top50 ∪ CF top50, emb+cf 가중치 tune
        union_grid  = [g for g in GRID if g[2] == 0]
        tune_scores = {w: evaluate(tune, w, 'union')[10][1] for w in union_grid}
        best_union  = max(tune_scores, key=tune_scores.get)
        report('retrieval 쿼리50∪CF50 + tuned emb+cf', evaluate(test, best_union, 'union'), fmt(best_union),
               diversity(test, best_union, 'union'))
        recall_union = np.mean([feats[i]['tpos_union'] >= 0 for i in test])
        rows.append({'protocol': proto, 'CF': name, 'mode': '후보 안에 정답 비율 (쿼리50∪CF50)', 'k': 10,
                     'HitRate': recall_union, 'NDCG': np.nan, 'MRR': np.nan})

        # 후보 수 실험: 쿼리 top-N → emb+cf 가중합 (N마다 w_emb tune). argsort 대신 target 순위만 계산
        def size_eval(idx_range, n, we):
            hits, ndcgs = [], []
            for i in idx_range:
                t = users[i]['tpos_big']
                if t < 0 or t >= n:
                    hits.append(0.0); ndcgs.append(0.0); continue
                s = we * normalize(users[i]['e_big'][:n]) + (1 - we) * normalize(c_big[i][:n])
                r = int((s > s[t]).sum())
                hits.append(float(r < 10)); ndcgs.append(1 / np.log2(r + 2) if r < 10 else 0.0)
            return np.mean(hits), np.mean(ndcgs)
        for n in SIZES:
            best_we = max([x / 10 for x in range(11)], key=lambda we: size_eval(tune, n, we)[1])
            hit, ndcg = size_eval(test, n, best_we)
            size_rows.append({'protocol': proto, 'CF': name, 'N': n, 'w(e/cf)': f'{best_we:g}/{1 - best_we:g}',
                              '후보 안 정답 비율': np.mean([0 <= users[i]['tpos_big'] < n for i in test]),
                              'HitRate@10': hit, 'NDCG@10': ndcg})

        # 유저별 결과 → 구간 분석 (정답 신상품 여부, 이력 길이)
        seg_modes = {'Embedding only': ((1, 0, 0), 'minmax'), '기존 app (.5/.3/.2)': ((0.5, 0.3, 0.2), 'minmax'),
                     '기존 hybrid (.3/.7/0)': ((0.3, 0.7, 0), 'minmax'),
                     'retrieval 쿼리50∪CF50': (best_union, 'union')}
        for mode, (w, fusion) in seg_modes.items():
            for i, (hit, ndcg, _) in per_user(test, w, fusion):
                seg_rows.append({'protocol': proto, 'CF': name, 'mode': mode, 'cold': users[i]['cold'],
                                 'hist': users[i]['hist'], 'HitRate@10': hit, 'NDCG@10': ndcg})

        # app epsilon 탐색: 기존 app 가중치 & 최적 가중치에 적용 (eps=0.1 → 5 seed 평균, eps=1.0 → 항상 교체)
        for base, (w, fusion) in {'기존 app': ((0.5, 0.3, 0.2), 'minmax'), 'best': (best_overall[1], best_overall[2])}.items():
            for eps in (EPSILON, 1.0):
                res = [evaluate(test, w, fusion, eps=eps, seed=s) for s in range(5 if eps < 1 else 1)]
                report(f'{base} + explore eps={eps}', {k: np.mean([r[k] for r in res], axis=0) for k in K_LIST},
                       fmt(w), diversity(test, w, fusion, eps=eps))
    print(f"[{proto}] 임베딩 top50 안에 target이 들어온 비율(Recall@50, test 유저): {recall50:.4f}")

# ── 출력 ─────────────────────────────────────────────────
df  = pd.DataFrame(rows)
div = pd.DataFrame(div_rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
for proto in protocols:
    d = df[df.protocol == proto]
    print(f"\n######## 프로토콜: {proto} (test 유저 기준, 가중치는 tune 유저에서 선택) ########")
    for metric, k in (('HitRate', 5), ('HitRate', 10), ('NDCG', 10)):
        print(f"\n--- {metric}@{k} ---")
        print(d[d.k == k].pivot(index='mode', columns='CF', values=metric).round(4))
    print("\n--- 선택된 가중치 (emb/cf/bm25) ---")
    print(d[(d.k == 10) & d['mode'].str.startswith('tuned')].pivot(index='mode', columns='CF', values='weights(e/cf/bm)'))
    print("\n--- 다양성 (coverage=고유 아이템 비율 ↑, avg_log_pop=인기도 ↓면 덜 뻔함, cold_share=신상품 비율) ---")
    print(div[div.protocol == proto].drop(columns='protocol').set_index(['CF', 'mode']).round(4))

seg = pd.DataFrame(seg_rows)
for proto in protocols:
    s = seg[seg.protocol == proto]
    print(f"\n######## 구간 분석: {proto} — HitRate@10 / NDCG@10 ########")
    for col, label in (('cold', '정답이 신상품인가 (True=신상품)'), ('hist', '유저 train 이력 길이')):
        print(f"\n--- {label} ---")
        t = s.groupby(['mode', col, 'CF'])[['HitRate@10', 'NDCG@10']].mean().unstack('CF').round(4)
        print(t)
        print('유저 수:', s[(s.CF == s.CF.iloc[0]) & (s['mode'] == s['mode'].iloc[0])][col].value_counts().to_dict())
seg.to_csv('./hybrid_twotower_per_user.csv', index=False)

size = pd.DataFrame(size_rows)
for proto in protocols:
    print(f"\n######## 후보 수 실험: {proto} (쿼리 임베딩 top-N → emb+cf 가중합, N마다 가중치 tune) ########")
    s = size[size.protocol == proto]
    for col in ('후보 안 정답 비율', 'HitRate@10', 'NDCG@10', 'w(e/cf)'):
        print(f"\n--- {col} ---")
        print(s.pivot(index='N', columns='CF', values=col).round(4) if col != 'w(e/cf)' else s.pivot(index='N', columns='CF', values=col))
size.to_csv('./hybrid_candidate_size.csv', index=False)

print("\n######## 단계별 latency (ms/요청) ########")
for k, v in latency.items():
    print(f"{k:<36}{v:>10.1f}")

df.to_csv('./hybrid_twotower_results.csv', index=False)
div.to_csv('./hybrid_twotower_diversity.csv', index=False)
print("\n저장: ./hybrid_twotower_results.csv, ./hybrid_twotower_diversity.csv")
