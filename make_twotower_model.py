import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

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

articles  = pd.read_csv('./articles.csv', dtype={'article_id': str})
customers = pd.read_csv('./customers.csv', usecols=['customer_id', 'age', 'club_member_status'])

# ── 상수 ─────────────────────────────────────────────────
EMBED_DIM   = 64
BATCH_SIZE  = 4096
N_EPOCHS    = 3
LR          = 1e-3
TEMPERATURE = 0.05
ITEM_FEATS  = ['product_type_no', 'graphical_appearance_no', 'colour_group_code',
               'department_no', 'index_group_no', 'garment_group_no']

# ── 매핑 ─────────────────────────────────────────────────
train_interaction = train_df[['customer_id', 'article_id']].drop_duplicates()

iid_map  = {a: i for i, a in enumerate(train_interaction['article_id'].unique())}
uid_map  = {u: i for i, u in enumerate(train_interaction['customer_id'].unique())}
idx2item = {v: k for k, v in iid_map.items()}
print(f"유저: {len(uid_map)} | 아이템: {len(iid_map)} | interaction: {len(train_interaction)}")

# 아이템 피처: 카테고리 컬럼마다 0부터 인덱스 재부여
art = articles.set_index('article_id').loc[list(iid_map.keys()), ITEM_FEATS]
item_feat_np, item_feat_sizes = [], []
for col in ITEM_FEATS:
    codes, uniques = pd.factorize(art[col])
    item_feat_np.append(codes)
    item_feat_sizes.append(len(uniques))
item_feat_np = np.stack(item_feat_np, axis=1)                     # (n_items, n_feats)

# 유저 피처: 나이 버킷(결측=0), 멤버십 상태
cust = customers.set_index('customer_id').reindex(list(uid_map.keys()))
age_bucket = (cust['age'].fillna(0) // 5).clip(0, 20).astype(int).values
club, _    = pd.factorize(cust['club_member_status'].fillna('NONE'))
user_feat_np = np.stack([age_bucket, club], axis=1)
user_feat_sizes = [21, int(club.max()) + 1]

# ── 모델 ─────────────────────────────────────────────────
class Tower(nn.Module):
    def __init__(self, n_ids, feat_sizes, dim=EMBED_DIM):
        super().__init__()
        self.id_emb    = nn.Embedding(n_ids, dim)
        self.feat_embs = nn.ModuleList([nn.Embedding(n, dim // 4) for n in feat_sizes])
        in_dim = dim + (dim // 4) * len(feat_sizes)
        self.mlp = nn.Sequential(nn.Linear(in_dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, ids, feats):
        x = [self.id_emb(ids)] + [emb(feats[:, i]) for i, emb in enumerate(self.feat_embs)]
        return F.normalize(self.mlp(torch.cat(x, dim=-1)), dim=-1)

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, user_feat_sizes, item_feat_sizes):
        super().__init__()
        self.user_tower = Tower(n_users, user_feat_sizes)
        self.item_tower = Tower(n_items, item_feat_sizes)

# ── 학습 (in-batch negative softmax) ─────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TwoTower(len(uid_map), len(iid_map), user_feat_sizes, item_feat_sizes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

user_idx  = torch.tensor(train_interaction['customer_id'].map(uid_map).values, device=device)
item_idx  = torch.tensor(train_interaction['article_id'].map(iid_map).values, device=device)
user_feat = torch.tensor(user_feat_np, dtype=torch.long, device=device)
item_feat = torch.tensor(item_feat_np, dtype=torch.long, device=device)

# 인기 아이템이 negative로 과도하게 뽑히는 것 보정 (logQ correction)
item_count = torch.bincount(item_idx, minlength=len(iid_map)).float()
log_q      = torch.log(item_count / item_count.sum())

n = len(user_idx)
print(f"학습 시작 | device: {device}")
for epoch in range(N_EPOCHS):
    model.train()
    perm = torch.randperm(n, device=device)
    total_loss, n_batches = 0.0, 0
    for s in tqdm(range(0, n, BATCH_SIZE), desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
        b = perm[s:s + BATCH_SIZE]
        u, i = user_idx[b], item_idx[b]
        u_emb = model.user_tower(u, user_feat[u])
        i_emb = model.item_tower(i, item_feat[i])

        logits = u_emb @ i_emb.T / TEMPERATURE - log_q[i].unsqueeze(0)
        # 같은 배치 안에 같은 아이템이 여러 번 있으면 false negative → 마스킹
        same_item = (i.unsqueeze(1) == i.unsqueeze(0)) & ~torch.eye(len(b), dtype=torch.bool, device=device)
        logits = logits.masked_fill(same_item, -1e9)
        loss = F.cross_entropy(logits, torch.arange(len(b), device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    tqdm.write(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {total_loss/n_batches:.4f}")

print("학습 완료")

# ── 임베딩 추출 ──────────────────────────────────────────
@torch.no_grad()
def encode(tower, n_ids, feats, bs=65536):
    tower.eval()
    out = []
    for s in range(0, n_ids, bs):
        ids = torch.arange(s, min(s + bs, n_ids), device=device)
        out.append(tower(ids, feats[ids]))
    return torch.cat(out)

item_emb = encode(model.item_tower, len(iid_map), item_feat)
user_emb = encode(model.user_tower, len(uid_map), user_feat)

# ── 간단 평가: test 기간 구매 유저 Recall@12 ─────────────
test_truth = (
    test_df[test_df['customer_id'].isin(uid_map) & test_df['article_id'].isin(iid_map)]
    .groupby('customer_id')['article_id'].apply(set)
)
eval_users = test_truth.sample(min(10000, len(test_truth)), random_state=42)
recalls = []
with torch.no_grad():
    for s in range(0, len(eval_users), 1024):
        chunk = eval_users.iloc[s:s + 1024]
        u = torch.tensor([uid_map[c] for c in chunk.index], device=device)
        top = (user_emb[u] @ item_emb.T).topk(12, dim=-1).indices.cpu().numpy()
        for truth, rec in zip(chunk.values, top):
            hit = len(truth & {idx2item[r] for r in rec})
            recalls.append(hit / min(len(truth), 12))
print(f"Recall@12 ({len(recalls)} users): {np.mean(recalls):.4f}")

# ── 저장 ─────────────────────────────────────────────────
with open('./twotower_model.pkl', 'wb') as f:
    pickle.dump({
        'model':     model.cpu().state_dict(),
        'item_emb':  item_emb.cpu().numpy(),
        'user_emb':  user_emb.cpu().numpy(),
        'iid_map':   iid_map,
        'uid_map':   uid_map,
        'idx2item':  idx2item,
    }, f)
print("저장 완료: ./twotower_model.pkl")
