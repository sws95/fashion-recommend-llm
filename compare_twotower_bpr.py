import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Two-Tower vs BPR vs 인기순 — 같은 split / 같은 아이템 / 같은 평가 유저로 비교
# 먼저 make_twotower_model.py 실행해서 twotower_model.pkl 생성 필요

# ── 데이터 로딩 ──────────────────────────────────────────
print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str, 'article_id': str
}, usecols=['t_dat', 'customer_id', 'article_id'])
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].quantile(0.8)

train_df = transactions[transactions['t_dat'] < split_date]
test_df  = transactions[transactions['t_dat'] >= split_date]
del transactions

with open('./twotower_model.pkl', 'rb') as f:
    tt = pickle.load(f)
uid_map, iid_map, idx2item = tt['uid_map'], tt['iid_map'], tt['idx2item']
n_users, n_items = len(uid_map), len(iid_map)

train_interaction = train_df[['customer_id', 'article_id']].drop_duplicates()

# ── 상수 ─────────────────────────────────────────────────
EMBED_DIM  = 64
BATCH_SIZE = 8192
N_EPOCHS   = 5
LR         = 1e-3
L2_REG     = 1e-5
K          = 12
N_EVAL     = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_idx = torch.tensor(train_interaction['customer_id'].map(uid_map).values, device=device)
item_idx = torch.tensor(train_interaction['article_id'].map(iid_map).values, device=device)

# ── BPR 학습 (MF + uniform negative) ─────────────────────
user_e = torch.nn.Embedding(n_users, EMBED_DIM).to(device)
item_e = torch.nn.Embedding(n_items, EMBED_DIM).to(device)
torch.nn.init.normal_(user_e.weight, std=0.01)
torch.nn.init.normal_(item_e.weight, std=0.01)
optimizer = torch.optim.Adam(list(user_e.parameters()) + list(item_e.parameters()), lr=LR)

n = len(user_idx)
print(f"BPR 학습 시작 | device: {device}")
for epoch in range(N_EPOCHS):
    perm = torch.randperm(n, device=device)
    total_loss, n_batches = 0.0, 0
    for s in tqdm(range(0, n, BATCH_SIZE), desc=f"BPR Epoch {epoch+1}/{N_EPOCHS}"):
        b = perm[s:s + BATCH_SIZE]
        u, p = user_idx[b], item_idx[b]
        q = torch.randint(0, n_items, (len(b),), device=device)
        ue, pe, qe = user_e(u), item_e(p), item_e(q)
        loss = -F.logsigmoid((ue * pe).sum(-1) - (ue * qe).sum(-1)).mean()
        loss = loss + L2_REG * (ue.pow(2).sum() + pe.pow(2).sum() + qe.pow(2).sum()) / len(b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    tqdm.write(f"BPR Epoch {epoch+1}/{N_EPOCHS} | Loss: {total_loss/n_batches:.4f}")

with open('./bpr_torch_model.pkl', 'wb') as f:
    pickle.dump({'user_emb': user_e.weight.detach().cpu().numpy(),
                 'item_emb': item_e.weight.detach().cpu().numpy(),
                 'uid_map': uid_map, 'idx2item': idx2item}, f)

# ── 평가 (article_id 기준, 신상품 구매도 정답에 포함) ────
test_truth = (
    test_df[test_df['customer_id'].isin(uid_map)]
    .groupby('customer_id')['article_id'].apply(set)
)
eval_users = test_truth.sample(min(N_EVAL, len(test_truth)), random_state=42)
train_items = set(iid_map)

# 구간: train 구매 수(유저 이력 길이)
n_hist = train_interaction.groupby('customer_id').size()
hist_bucket = pd.cut(n_hist.reindex(eval_users.index), [0, 5, 20, np.inf], labels=['1-5', '6-20', '21+'])

# 인기순: train 마지막 1주 판매량 기준
last_week = train_df[train_df['t_dat'] >= train_df['t_dat'].max() - pd.Timedelta(days=7)]
pop_top = last_week['article_id'].value_counts().index[:K].tolist()

# name → (유저 임베딩, 유저 id→행, 아이템 임베딩, 행→article_id)
models = {
    'TwoTower': (torch.tensor(tt['user_emb'], device=device), {u: i for u, i in uid_map.items()},
                 torch.tensor(tt['item_emb'], device=device), [idx2item[i] for i in range(n_items)]),
    'BPR':      (user_e.weight.detach(), {u: i for u, i in uid_map.items()},
                 item_e.weight.detach(), [idx2item[i] for i in range(n_items)]),
}
try:
    with open('./twotower_qwen_model.pkl', 'rb') as f:
        tq = pickle.load(f)
    models['TwoTower+Qwen'] = (torch.tensor(tq['user_emb'], device=device),
                               {u: i for i, u in enumerate(tq['user_ids'])},
                               torch.tensor(tq['item_emb'], device=device), tq['item_ids'])
except FileNotFoundError:
    print("twotower_qwen_model.pkl 없음 → TwoTower+Qwen 제외")
# 앱에서 쓰는 cornac BPR (make_bpr_model.py, 100 iter). score = i_bias + u·i → [u, 1]·[i, b]
try:
    with open('./bpr_model.pkl', 'rb') as f:
        cb = pickle.load(f)
    cm, cds = cb['model'], cb['dataset']
    c_idx2item = {v: k for k, v in cds.iid_map.items()}
    models['BPR (cornac, 앱)'] = (
        torch.tensor(np.hstack([cm.u_factors, np.ones((cm.u_factors.shape[0], 1))]), dtype=torch.float32, device=device),
        dict(cds.uid_map),
        torch.tensor(np.hstack([cm.i_factors, cm.i_biases.reshape(-1, 1)]), dtype=torch.float32, device=device),
        [c_idx2item[i] for i in range(len(c_idx2item))])
    del cb
except FileNotFoundError:
    print("bpr_model.pkl 없음 → cornac BPR 제외")

def metrics(rec, truth):
    hits = [1 if r in truth else 0 for r in rec]
    dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(truth), K)))
    cold = truth - train_items                   # test에 처음 등장한 신상품
    cold_hit = len(cold & set(rec)) / min(len(cold), K) if cold else np.nan
    return sum(hits) / min(len(truth), K), float(any(hits)), dcg / idcg, cold_hit

results = {}
with torch.no_grad():
    for name, (U, umap, I, item_ids) in models.items():
        scores = []
        for s in range(0, len(eval_users), 1024):
            chunk = eval_users.iloc[s:s + 1024]
            u = torch.tensor([umap[c] for c in chunk.index], device=device)
            top = (U[u] @ I.T).topk(K, dim=-1).indices.cpu().numpy()
            scores += [metrics([item_ids[r] for r in row], t) for row, t in zip(top, chunk.values)]
        results[name] = pd.DataFrame(scores, index=eval_users.index, columns=['Recall', 'HitRate', 'NDCG', 'ColdRecall'])
results['Popularity'] = pd.DataFrame([metrics(pop_top, t) for t in eval_users.values],
                                     index=eval_users.index, columns=['Recall', 'HitRate', 'NDCG', 'ColdRecall'])

# 재구매 baseline: 유저가 최근 산 아이템 12개 그대로 (app은 이미 산 아이템을 제외하므로 이 신호를 버리고 있음)
recent = (
    train_df[train_df['customer_id'].isin(eval_users.index)].sort_values('t_dat', ascending=False)
    .drop_duplicates(['customer_id', 'article_id']).groupby('customer_id')['article_id'].apply(lambda s: s.tolist()[:K])
)
results['RecentRepurchase'] = pd.DataFrame([metrics(recent[u] + [a for a in pop_top if a not in recent[u]][:K - len(recent[u])], t)
                                            for u, t in eval_users.items()],
                                           index=eval_users.index, columns=['Recall', 'HitRate', 'NDCG', 'ColdRecall'])
train_user_items_eval = (train_interaction[train_interaction['customer_id'].isin(eval_users.index)]
                         .groupby('customer_id')['article_id'].apply(set))
repurchase_share = np.mean([len(t & set(train_user_items_eval[u])) / len(t) for u, t in eval_users.items()])
print(f"test 구매 중 이미 train에서 산 아이템(재구매) 비율: {repurchase_share:.4f}")

print(f"\n=== {len(eval_users)} users, test 기간 구매 예측 @ {K} ===")
print("ColdRecall = test에서 신상품(train에 없던 아이템)을 산 유저 대상, 신상품 Recall")
print(pd.DataFrame({name: df.mean() for name, df in results.items()}).T.round(4))

print(f"\n=== 유저 이력 길이별 Recall@{K} ===")
print(pd.DataFrame({name: df['Recall'].groupby(hist_bucket, observed=True).mean()
                    for name, df in results.items()}).T.round(4))
print(hist_bucket.value_counts().sort_index().to_dict())
