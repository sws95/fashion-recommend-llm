import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# 투타워 + Qwen 임베딩
# - 아이템 타워: Qwen 텍스트 임베딩(고정) + 카테고리 + ID(드롭아웃 → 신상품은 ID 없이 콘텐츠로 임베딩)
# - 유저 타워: 최근 구매 아이템 Qwen 임베딩 평균 + 나이/멤버십 + ID(드롭아웃)
# 먼저 make_qwen_text_emb.py 실행 필요

# ── 데이터 로딩 ──────────────────────────────────────────
print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str, 'article_id': str
}, usecols=['t_dat', 'customer_id', 'article_id'])
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].quantile(0.8)

train_df = transactions[transactions['t_dat'] < split_date]
del transactions

articles  = pd.read_csv('./articles.csv', dtype={'article_id': str})
customers = pd.read_csv('./customers.csv', usecols=['customer_id', 'age', 'club_member_status'])
qwen_emb  = np.load('./qwen_text_emb.npy')
qwen_ids  = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id']
assert (qwen_ids.values == articles['article_id'].values).all()

# ── 상수 ─────────────────────────────────────────────────
EMBED_DIM   = 64
BATCH_SIZE  = 4096
N_EPOCHS    = 3
LR          = 1e-3
TEMPERATURE = 0.05
ID_DROPOUT  = 0.3
HIST_LEN    = 20
ITEM_FEATS  = ['product_type_no', 'graphical_appearance_no', 'colour_group_code',
               'department_no', 'index_group_no', 'garment_group_no']

# ── 매핑 (index 0 = 패딩/unknown) ────────────────────────
iid_map  = {a: i + 1 for i, a in enumerate(articles['article_id'])}     # 전체 아이템 (신상품 포함)
idx2item = {v: k for k, v in iid_map.items()}
train_items = set(train_df['article_id'].unique())
train_interaction = train_df[['customer_id', 'article_id']].drop_duplicates()
uid_map  = {u: i + 1 for i, u in enumerate(train_interaction['customer_id'].unique())}
n_items, n_users = len(iid_map) + 1, len(uid_map) + 1
print(f"유저: {len(uid_map)} | 아이템: {len(iid_map)} (train 등장 {len(train_items)}) | interaction: {len(train_interaction)}")

item_feat_np = np.zeros((n_items, len(ITEM_FEATS)), dtype=np.int64)
item_feat_sizes = []
for j, col in enumerate(ITEM_FEATS):
    codes, uniques = pd.factorize(articles[col])
    item_feat_np[1:, j] = codes
    item_feat_sizes.append(len(uniques))

cust = customers.set_index('customer_id').reindex(list(uid_map.keys()))
user_feat_np = np.zeros((n_users, 2), dtype=np.int64)
user_feat_np[1:, 0] = (cust['age'].fillna(0) // 5).clip(0, 20).astype(int).values
user_feat_np[1:, 1] = pd.factorize(cust['club_member_status'].fillna('NONE'))[0]
user_feat_sizes = [21, int(user_feat_np[:, 1].max()) + 1]

# 유저별 최근 구매 HIST_LEN개 (오른쪽 정렬, 0 패딩)
print("구매 이력 생성...")
recent = (
    train_df.sort_values('t_dat').drop_duplicates(['customer_id', 'article_id'], keep='last')
    .groupby('customer_id').tail(HIST_LEN)
)
recent_u = recent['customer_id'].map(uid_map).values
recent_i = recent['article_id'].map(iid_map).values
pos_in_user = recent.groupby('customer_id').cumcount(ascending=False).values   # 최신 = 0
user_hist_np = np.zeros((n_users, HIST_LEN), dtype=np.int64)
user_hist_np[recent_u, HIST_LEN - 1 - pos_in_user] = recent_i
del recent

# ── 모델 ─────────────────────────────────────────────────
class Tower(nn.Module):
    def __init__(self, n_ids, feat_sizes, content_dim, dim=EMBED_DIM):
        super().__init__()
        self.id_emb       = nn.Embedding(n_ids, dim, padding_idx=0)
        self.feat_embs    = nn.ModuleList([nn.Embedding(n, dim // 4) for n in feat_sizes])
        self.content_proj = nn.Linear(content_dim, dim)
        in_dim = dim * 2 + (dim // 4) * len(feat_sizes)
        self.mlp = nn.Sequential(nn.Linear(in_dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, ids, feats, content):
        if self.training:
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < ID_DROPOUT, 0)
        x = [self.id_emb(ids), self.content_proj(content)] + \
            [emb(feats[:, i]) for i, emb in enumerate(self.feat_embs)]
        return F.normalize(self.mlp(torch.cat(x, dim=-1)), dim=-1)

class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_tower = Tower(n_users, user_feat_sizes, qwen_emb.shape[1])
        self.item_tower = Tower(n_items, item_feat_sizes, qwen_emb.shape[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TwoTower().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

qwen      = torch.cat([torch.zeros(1, qwen_emb.shape[1], dtype=torch.float16),
                       torch.tensor(qwen_emb)]).to(device)                 # (n_items, 2048), 0행 = 패딩
user_hist = torch.tensor(user_hist_np, device=device)
user_feat = torch.tensor(user_feat_np, device=device)
item_feat = torch.tensor(item_feat_np, device=device)
user_idx  = torch.tensor(train_interaction['customer_id'].map(uid_map).values, device=device)
item_idx  = torch.tensor(train_interaction['article_id'].map(iid_map).values, device=device)

def hist_content(u, exclude=None):
    hist = user_hist[u]
    mask = hist > 0
    if exclude is not None:                      # 학습 시 정답 아이템은 이력에서 제외 (누수 방지)
        mask &= hist != exclude.unsqueeze(1)
    summed = (qwen[hist].float() * mask.unsqueeze(-1)).sum(1)
    return summed / mask.sum(1, keepdim=True).clamp(min=1)

item_count = torch.bincount(item_idx, minlength=n_items).float()
log_q      = torch.log((item_count + 1) / (item_count + 1).sum())

# ── 학습 ─────────────────────────────────────────────────
n = len(user_idx)
print(f"학습 시작 | device: {device}")
for epoch in range(N_EPOCHS):
    model.train()
    perm = torch.randperm(n, device=device)
    total_loss, n_batches = 0.0, 0
    for s in tqdm(range(0, n, BATCH_SIZE), desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
        b = perm[s:s + BATCH_SIZE]
        u, i = user_idx[b], item_idx[b]
        u_emb = model.user_tower(u, user_feat[u], hist_content(u, exclude=i))
        i_emb = model.item_tower(i, item_feat[i], qwen[i].float())

        logits = u_emb @ i_emb.T / TEMPERATURE - log_q[i].unsqueeze(0)
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
torch.save(model.state_dict(), './twotower_qwen_state.pt')     # 추출 단계 실패 대비

# ── 임베딩 추출 (신상품은 ID=0 → 콘텐츠만으로 임베딩) ──
model.eval()
CHUNK = 4096                                                  # 유저: CHUNK × HIST_LEN × 2048 float → 8GB GPU 기준
is_train_item = torch.tensor([idx2item.get(k) in train_items for k in range(n_items)], device=device)
with torch.no_grad():
    item_emb = []
    for s in range(1, n_items, 65536):
        ids = torch.arange(s, min(s + 65536, n_items), device=device)
        item_emb.append(model.item_tower(ids * is_train_item[ids], item_feat[ids], qwen[ids].float()))
    item_emb = torch.cat(item_emb)

    user_emb = []
    for s in range(1, n_users, CHUNK):
        ids = torch.arange(s, min(s + CHUNK, n_users), device=device)
        user_emb.append(model.user_tower(ids, user_feat[ids], hist_content(ids)))
    user_emb = torch.cat(user_emb)

# ── 저장 (임베딩 행 r ↔ index r+1) ───────────────────────
with open('./twotower_qwen_model.pkl', 'wb') as f:
    pickle.dump({
        'model':     model.cpu().state_dict(),
        'item_emb':  item_emb.cpu().numpy(),
        'user_emb':  user_emb.cpu().numpy(),
        'item_ids':  [idx2item[k] for k in range(1, n_items)],
        'user_ids':  list(uid_map.keys()),
    }, f)
print("저장 완료: ./twotower_qwen_model.pkl")
