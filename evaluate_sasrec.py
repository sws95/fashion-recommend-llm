import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sasrec import SASRec

# SASRec 검증 (연관 추천 /recommend/related 에서 사용)
# 1) sasrec.py 버그: 시퀀스를 왼쪽 패딩하는데 last_idx = (seq != 0).sum() - 1 로 뽑음
#    → 이력 < MAX_SEQ_LEN 유저는 마지막 아이템이 아닌 패딩 위치 출력을 씀. 수정판(x[:, -1])과 비교
# 2) 평가 2종
#    readme: make_sasrec_model.py 방식 — 학습 target(시퀀스 마지막)을 그대로 평가, 1 pos + 99 random neg
#    future: train 전체 시퀀스 입력 → test 기간 첫 신규 구매 예측, 전체 아이템 랭킹 (학습 때 못 본 정답)
#    future에서 BPR / TwoTower / TwoTower+Qwen / 인기순과 같은 유저로 비교
# 학습 설정은 make_sasrec_model.py와 동일, epoch만 축소

# ── 데이터 ───────────────────────────────────────────────
print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str, 'article_id': str
}, usecols=['t_dat', 'customer_id', 'article_id'])
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].quantile(0.8)
train_df = transactions[transactions['t_dat'] < split_date]
test_df  = transactions[transactions['t_dat'] >= split_date]
del transactions

user_seq = train_df.sort_values('t_dat').groupby('customer_id')['article_id'].apply(list).to_dict()
user_seq = {u: s for u, s in user_seq.items() if len(s) >= 2}
all_items = train_df['article_id'].unique()
iid_map   = {a: i + 1 for i, a in enumerate(all_items)}
idx2item  = {v: k for k, v in iid_map.items()}
n_items   = len(iid_map) + 1
uids      = list(user_seq.keys())
seqs      = [[iid_map[a] for a in user_seq[u]] for u in tqdm(uids, desc="시퀀스 인덱스 변환")]
print(f"유저: {len(uids)} | 아이템: {n_items}")

MAX_SEQ_LEN = 50
EMBED_DIM   = 64
N_HEADS     = 2
N_LAYERS    = 2
DROPOUT     = 0.2
BATCH_SIZE  = 256
N_EPOCHS    = 15            # 원본 100 → 시간상 축소
LR          = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad(s):
    s = s[-MAX_SEQ_LEN:]
    return [0] * (MAX_SEQ_LEN - len(s)) + s

train_in  = torch.tensor([pad(s[:-1]) for s in seqs], device=device)
train_pos = torch.tensor([s[-1] for s in seqs], device=device)
full_in   = torch.tensor([pad(s) for s in seqs], device=device)       # future 평가용 (train 전체)

class SASRecFixed(SASRec):
    """수정 2가지
    1) 왼쪽 패딩 → 마지막 아이템은 항상 맨 끝 (x[:, -1])
    2) 원본은 패딩 위치가 attend할 key가 하나도 없어 eval 모드(fast path)에서 NaN → 다음 layer에서 시퀀스 전체로 전파.
       causal + key padding을 합친 mask에서 대각선(자기 자신)은 항상 허용해 NaN 행이 생기지 않게 함"""
    def forward(self, seq):
        batch_size, seq_len = seq.shape
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x   = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool), diagonal=1)
        mask   = causal.unsqueeze(0) | (seq == 0).unsqueeze(1)                     # (B, L, L) True = 차단
        mask   = mask & ~torch.eye(seq_len, device=seq.device, dtype=torch.bool)
        n_heads = self.transformer.layers[0].self_attn.num_heads
        x = self.transformer(x, mask=mask.repeat_interleave(n_heads, dim=0))
        return self.norm(x)[:, -1]

def train(model_cls, name):
    torch.manual_seed(42)
    model = model_cls(n_items, EMBED_DIM, N_HEADS, N_LAYERS, MAX_SEQ_LEN, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(train_in)
    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        total, nb, nan_batches = 0.0, 0, 0
        for s in tqdm(range(0, n, BATCH_SIZE), desc=f"{name} {epoch+1}/{N_EPOCHS}", mininterval=10):
            b   = perm[s:s + BATCH_SIZE]
            pos = train_pos[b]
            neg = torch.randint(1, n_items, pos.shape, device=device)
            u   = model(train_in[b])
            loss = -torch.log(torch.sigmoid((u * model.item_emb(pos)).sum(-1) - (u * model.item_emb(neg)).sum(-1)) + 1e-8).mean()
            if torch.isnan(loss):
                nan_batches += 1
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        tqdm.write(f"{name} Epoch {epoch+1} | Loss: {total/max(nb,1):.4f} | NaN 배치: {nan_batches}")
    model.eval()
    return model

@torch.no_grad()
def user_vectors(model, inp, rows):
    out = []
    for s in range(0, len(rows), 4096):
        out.append(model(inp[rows[s:s + 4096]]))
    out = torch.cat(out)
    print(f"  NaN 유저 벡터 비율: {torch.isnan(out).any(1).float().mean():.4f}")
    return torch.nan_to_num(out)                      # NaN → 0 (모든 점수 동점 → 아래에서 오답 처리)

def metrics_from_rank(rank):
    return {k: ((rank < k).mean(), np.where(rank < k, 1 / np.log2(rank + 2), 0).mean()) for k in (5, 10)}

rng = np.random.RandomState(42)
results = []

# ── 평가 유저 ────────────────────────────────────────────
readme_rows = torch.tensor(rng.choice(len(uids), 10000, replace=False), device=device)

uid_row  = {u: i for i, u in enumerate(uids)}
bought   = {u: set(user_seq[u]) for u in uids}
first_new = (
    test_df[test_df['customer_id'].isin(uid_row) & test_df['article_id'].isin(iid_map)]
    .sort_values('t_dat')
)
first_new = first_new[[a not in bought[u] for u, a in zip(first_new['customer_id'], first_new['article_id'])]]
first_new = first_new.drop_duplicates('customer_id')
first_new = first_new.sample(min(10000, len(first_new)), random_state=42)
fut_rows   = torch.tensor([uid_row[u] for u in first_new['customer_id']], device=device)
fut_target = torch.tensor([iid_map[a] for a in first_new['article_id']], device=device)
print(f"future 평가 유저: {len(fut_rows)}")

def eval_future(name, U, I, item_index_of):
    """U: (n_eval, d) 유저 벡터, I: (n_items_model, d), item_index_of: SASRec idx → 모델 아이템 행 (없으면 -1)"""
    ranks = []
    for s in range(0, len(U), 1024):
        sc = U[s:s + 1024] @ I.T
        rows = fut_rows[s:s + 1024]
        # 이미 산 아이템 제외 (app과 동일)
        for j, r in enumerate(rows.tolist()):
            bought_idx = item_index_of[torch.tensor([iid_map[a] for a in bought[uids[r]]], device=device)]
            sc[j, bought_idx[bought_idx >= 0]] = -1e9
        t_idx = item_index_of[fut_target[s:s + 1024]]
        t_sc  = sc.gather(1, t_idx.clamp(min=0).unsqueeze(1))
        rank  = (sc >= t_sc).sum(1) - 1               # 동점은 불리하게 (자기 자신 1개 제외)
        rank[t_idx < 0] = 10 ** 9
        ranks.append(rank.cpu().numpy())
    m = metrics_from_rank(np.concatenate(ranks))
    results.append({'eval': 'future (전체 랭킹)', 'model': name,
                    **{f'{n}@{k}': v[i] for k, v in m.items() for i, n in enumerate(['HitRate', 'NDCG'])}})

# ── SASRec 원본 vs 수정 ──────────────────────────────────
identity = torch.arange(n_items, device=device)
for cls, name in ((SASRec, 'SASRec (원본)'), (SASRecFixed, 'SASRec (last_idx 수정)')):
    model = train(cls, name)
    with torch.no_grad():
        # readme: 1 pos + 99 neg
        U = user_vectors(model, train_in, readme_rows)
        pos = train_pos[readme_rows]
        neg = torch.randint(1, n_items, (len(pos), 99), device=device)
        cand = torch.cat([pos.unsqueeze(1), neg], 1)
        sc = (U.unsqueeze(1) * model.item_emb(cand)).sum(-1)
        rank = (sc[:, 1:] >= sc[:, :1]).sum(1).cpu().numpy()   # 동점은 불리하게
        m = metrics_from_rank(rank)
        results.append({'eval': 'readme (1+99, 학습 target)', 'model': name,
                        **{f'{n}@{k}': v[i] for k, v in m.items() for i, n in enumerate(['HitRate', 'NDCG'])}})
        # future: 전체 아이템
        eval_future(name, user_vectors(model, full_in, fut_rows), model.item_emb.weight[1:], identity - 1)
    del model
    torch.cuda.empty_cache()

# ── 같은 future 유저로 CF 모델 비교 ──────────────────────
def mapped(row2item):
    pos = {a: i for i, a in enumerate(row2item)}
    return torch.tensor([-1] + [pos.get(idx2item[i], -1) for i in range(1, n_items)], device=device)

with open('./bpr_torch_model.pkl', 'rb') as f:
    d = pickle.load(f)
umap = d['uid_map']
ok = torch.tensor([uids[r] in umap for r in fut_rows.tolist()])
print(f"CF 모델 유저 커버리지: {ok.float().mean():.4f}")
for path, name in (('./bpr_torch_model.pkl', 'BPR'), ('./twotower_model.pkl', 'TwoTower'), ('./twotower_qwen_model.pkl', 'TwoTower+Qwen')):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    if 'user_ids' in d:
        umap, row2item = {u: i for i, u in enumerate(d['user_ids'])}, d['item_ids']
    else:
        umap, row2item = d['uid_map'], [d['idx2item'][i] for i in range(len(d['idx2item']))]
    U = torch.tensor(d['user_emb'], device=device)[torch.tensor([umap[uids[r]] for r in fut_rows.tolist()], device=device)]
    eval_future(name, U, torch.tensor(d['item_emb'], device=device), mapped(row2item))

# 인기순: train 마지막 1주
last_week = train_df[train_df['t_dat'] >= train_df['t_dat'].max() - pd.Timedelta(days=7)]
pop = last_week['article_id'].value_counts()
pop_vec = torch.zeros(n_items - 1, 1, device=device)
for a, c in pop.items():
    if a in iid_map:
        pop_vec[iid_map[a] - 1, 0] = float(c)
eval_future('Popularity (최근 1주)', torch.ones(len(fut_rows), 1, device=device), pop_vec, identity - 1)

df = pd.DataFrame(results)
pd.set_option('display.width', 200)
print("\n=== SASRec / 다음 신규 구매 예측 ===")
print(df.round(4).to_string(index=False))
df.to_csv('./sasrec_eval_results.csv', index=False)
print("\n저장: ./sasrec_eval_results.csv")
