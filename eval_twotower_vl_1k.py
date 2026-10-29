import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import chromadb
from tqdm import tqdm

# Two-Tower 아이템 타워: 텍스트 Qwen 임베딩 vs VL(이미지+텍스트) Qwen 임베딩
# VL 임베딩이 있는 상품이 1,000개(ChromaDB fashion_items)뿐 → 1,000개 카탈로그로 따로 학습해 비교
# 구조·하이퍼파라미터는 make_twotower_qwen_model.py와 동일, seed 고정. 콘텐츠 입력만 다름
# 평가: test 기간에 이 1,000개 중 구매한 유저 → 1,000개 랭킹 Recall@12 / HitRate@12

EMBED_DIM, BATCH_SIZE, N_EPOCHS, LR, TEMPERATURE, ID_DROPOUT, HIST_LEN, K = 64, 2048, 5, 1e-3, 0.05, 0.3, 20, 12
ITEM_FEATS = ['product_type_no', 'graphical_appearance_no', 'colour_group_code',
              'department_no', 'index_group_no', 'garment_group_no']
device = torch.device("cuda")

# ── 콘텐츠 임베딩 2종 ────────────────────────────────────
r = chromadb.PersistentClient(path='./chroma_db').get_collection('fashion_items').get(include=['embeddings'])
items = r['ids']
vl = np.array(r['embeddings'], dtype=np.float32)
all_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
pos = {a: i for i, a in enumerate(all_ids)}
txt = np.load('./qwen_text_emb.npy')[[pos[a] for a in items]].astype(np.float32)
iid_map = {a: i + 1 for i, a in enumerate(items)}           # 0 = 패딩
n_items = len(items) + 1

# ── 데이터 (1,000개 상품 거래만) ─────────────────────────
print("데이터 로딩...")
tr = pd.read_csv('./transactions_train.csv', dtype={'customer_id': str, 'article_id': str},
                 usecols=['t_dat', 'customer_id', 'article_id'])
tr['t_dat'] = pd.to_datetime(tr['t_dat'])
split_date = tr['t_dat'].quantile(0.8)                       # 전체 기준 split (본 실험과 동일)
tr = tr[tr['article_id'].isin(iid_map)]
train_df, test_df = tr[tr['t_dat'] < split_date], tr[tr['t_dat'] >= split_date]
train_int = train_df[['customer_id', 'article_id']].drop_duplicates()
uid_map = {u: i + 1 for i, u in enumerate(train_int['customer_id'].unique())}
n_users = len(uid_map) + 1
print(f"유저 {len(uid_map)} | 아이템 {len(items)} | train interaction {len(train_int)}")

articles = pd.read_csv('./articles.csv', dtype={'article_id': str}).set_index('article_id').loc[items]
item_feat_np = np.zeros((n_items, len(ITEM_FEATS)), dtype=np.int64)
feat_sizes = []
for j, c in enumerate(ITEM_FEATS):
    codes, uniq = pd.factorize(articles[c]); item_feat_np[1:, j] = codes; feat_sizes.append(len(uniq))
cust = pd.read_csv('./customers.csv', usecols=['customer_id', 'age', 'club_member_status']).set_index('customer_id').reindex(list(uid_map))
user_feat_np = np.zeros((n_users, 2), dtype=np.int64)
user_feat_np[1:, 0] = (cust['age'].fillna(0) // 5).clip(0, 20).astype(int).values
user_feat_np[1:, 1] = pd.factorize(cust['club_member_status'].fillna('NONE'))[0]
user_sizes = [21, int(user_feat_np[:, 1].max()) + 1]

recent = (train_df.sort_values('t_dat').drop_duplicates(['customer_id', 'article_id'], keep='last')
          .groupby('customer_id').tail(HIST_LEN))
hist_np = np.zeros((n_users, HIST_LEN), dtype=np.int64)
hist_np[recent['customer_id'].map(uid_map).values,
        HIST_LEN - 1 - recent.groupby('customer_id').cumcount(ascending=False).values] = recent['article_id'].map(iid_map).values

truth = test_df[test_df['customer_id'].isin(uid_map)].groupby('customer_id')['article_id'].apply(lambda s: {iid_map[a] for a in s})
print(f"평가 유저 {len(truth)}")

U_FEAT, I_FEAT, HIST = (torch.tensor(x, device=device) for x in (user_feat_np, item_feat_np, hist_np))
U_IDX = torch.tensor(train_int['customer_id'].map(uid_map).values, device=device)
I_IDX = torch.tensor(train_int['article_id'].map(iid_map).values, device=device)
cnt = torch.bincount(I_IDX, minlength=n_items).float()
log_q = torch.log((cnt + 1) / (cnt + 1).sum())

class Tower(nn.Module):
    def __init__(self, n_ids, sizes, content_dim, dim=EMBED_DIM):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, dim, padding_idx=0)
        self.feat = nn.ModuleList([nn.Embedding(n, dim // 4) for n in sizes])
        self.proj = nn.Linear(content_dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim * 2 + dim // 4 * len(sizes), dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
    def forward(self, ids, feats, content):
        if self.training:
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < ID_DROPOUT, 0)
        x = [self.id_emb(ids), self.proj(content)] + [e(feats[:, i]) for i, e in enumerate(self.feat)]
        return F.normalize(self.mlp(torch.cat(x, -1)), dim=-1)

def run(name, content_np, eval_content_np=None):
    """content_np로 학습. eval_content_np가 있으면 추론 때만 그 임베딩으로 교체 (재학습 없음)"""
    torch.manual_seed(42); np.random.seed(42)
    C = torch.cat([torch.zeros(1, content_np.shape[1]), torch.tensor(content_np)]).to(device)
    C_eval = C if eval_content_np is None else torch.cat([torch.zeros(1, eval_content_np.shape[1]), torch.tensor(eval_content_np)]).to(device)
    ut, it = Tower(n_users, user_sizes, C.shape[1]).to(device), Tower(n_items, feat_sizes, C.shape[1]).to(device)
    opt = torch.optim.Adam(list(ut.parameters()) + list(it.parameters()), lr=LR)
    def hist_c(u, excl=None, C=C):
        h = HIST[u]; m = h > 0
        if excl is not None:
            m &= h != excl.unsqueeze(1)
        return (C[h] * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
    n = len(U_IDX)
    for ep in range(N_EPOCHS):
        ut.train(); it.train()
        perm, tot, nb = torch.randperm(n, device=device), 0.0, 0
        for s in range(0, n, BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]; u, i = U_IDX[b], I_IDX[b]
            logits = ut(u, U_FEAT[u], hist_c(u, i)) @ it(i, I_FEAT[i], C[i]).T / TEMPERATURE - log_q[i].unsqueeze(0)
            logits = logits.masked_fill((i.unsqueeze(1) == i.unsqueeze(0)) & ~torch.eye(len(b), dtype=torch.bool, device=device), -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  {name} epoch {ep + 1} loss {tot / nb:.4f}")
    ut.eval(); it.eval()
    with torch.no_grad():
        ids = torch.arange(1, n_items, device=device)
        I = it(ids, I_FEAT[ids], C_eval[ids])
        users = torch.tensor([uid_map[u] for u in truth.index], device=device)
        rec, hit = [], []
        for s in range(0, len(users), 8192):
            u = users[s:s + 8192]
            top = ((ut(u, U_FEAT[u], hist_c(u, C=C_eval)) @ I.T).topk(K).indices + 1).cpu().numpy()
            for row, t in zip(top, truth.values[s:s + 8192]):
                h = len(t & set(row)); rec.append(h / min(len(t), K)); hit.append(h > 0)
        # 유저 1,000명 하이브리드 평가(eval_user_hybrid_1k.py)용 벡터 저장
        all_u = torch.arange(1, n_users, device=device)
        U_all = torch.cat([ut(u, U_FEAT[u], hist_c(u, C=C_eval)) for u in all_u.split(65536)])
        np.savez(f'./tt1k_{name}.npz', U=U_all.cpu().numpy(), I=I.cpu().numpy(),
                 user_ids=np.array(list(uid_map)), item_ids=np.array(items))
    return np.mean(rec), np.mean(hit)

results = {'ID + 카테고리 (콘텐츠 없음)': run('ID', np.zeros_like(txt[:, :1])),
           'Two-Tower + 텍스트 Qwen': run('텍스트', txt),
           'Two-Tower + VL Qwen (이미지+텍스트)': run('VL', vl),
           '텍스트로 학습 → 추론만 VL (사진 추가)': run('텍스트→VL', txt, vl)}
pop = (train_df[train_df['t_dat'] >= train_df['t_dat'].max() - pd.Timedelta(days=7)]['article_id']
       .value_counts().index[:K].map(iid_map).tolist())
results['Popularity (최근 1주)'] = (np.mean([len(t & set(pop)) / min(len(t), K) for t in truth.values]),
                                   np.mean([len(t & set(pop)) > 0 for t in truth.values]))
print(f"\n=== 상품 1,000개 카탈로그, 평가 유저 {len(truth)}명, @{K} ===")
for k, (rc, ht) in results.items():
    print(f"{k:<36} Recall {rc:.4f}  HitRate {ht:.4f}")
