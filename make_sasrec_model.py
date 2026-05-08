import sys
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sasrec import SASRec

# ── 트랜잭션 로딩 ────────────────────────────────────────
print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str, 'article_id': str
})
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].quantile(0.8)

train_df = transactions[transactions['t_dat'] < split_date]
print(f"train: {len(train_df)}")

# ── 시퀀스 생성 ──────────────────────────────────────────
user_seq = (
    train_df.sort_values('t_dat')
    .groupby('customer_id')['article_id']
    .apply(list)
    .to_dict()
)
user_seq = {u: seq for u, seq in user_seq.items() if len(seq) >= 2}
print(f"시퀀스 2개 이상 유저: {len(user_seq)}")

# ── 아이템 매핑 ──────────────────────────────────────────
all_items = train_df['article_id'].unique()
iid_map   = {item: i + 1 for i, item in enumerate(all_items)}
idx2item  = {v: k for k, v in iid_map.items()}
n_items   = len(iid_map) + 1

uid_map  = {u: i for i, u in enumerate(user_seq.keys())}
print(f"유저: {len(uid_map)} | 아이템: {n_items}")

user_seq_idx = {
    uid_map[u]: [iid_map[a] for a in seq if a in iid_map]
    for u, seq in tqdm(user_seq.items(), desc="시퀀스 인덱스 변환")
}
user_target = {
    uid_map[u]: user_seq_idx[uid_map[u]][-1]
    for u in user_seq.keys()
}

# ── 상수 ─────────────────────────────────────────────────
MAX_SEQ_LEN = 50
EMBED_DIM   = 64
N_HEADS     = 2
N_LAYERS    = 2
DROPOUT     = 0.2
BATCH_SIZE  = 256
N_EPOCHS    = 100
LR          = 1e-3

# ── 데이터셋 ─────────────────────────────────────────────
class SASRecDataset(Dataset):
    def __init__(self, user_seq_idx, n_items, max_len=MAX_SEQ_LEN):
        self.data    = []
        self.n_items = n_items
        for user_idx, seq in tqdm(user_seq_idx.items(), desc="데이터셋 생성"):
            input_seq = seq[:-1][-max_len:]
            pad_len   = max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq
            self.data.append((user_idx, input_seq, seq[-1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, seq, target = self.data[idx]
        neg = np.random.randint(1, self.n_items)
        while neg == target:
            neg = np.random.randint(1, self.n_items)
        return (
            torch.tensor(seq,    dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
        )

dataset    = SASRecDataset(user_seq_idx, n_items)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ── 학습 ─────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = SASRec(n_items, EMBED_DIM, N_HEADS, N_LAYERS, MAX_SEQ_LEN, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(f"학습 시작 | device: {device}")

for epoch in tqdm(range(N_EPOCHS), desc="전체 진행"):
    model.train()
    total_loss = 0.0
    for seq, pos, neg in dataloader:
        seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
        user_emb  = model(seq)
        pos_score = (user_emb * model.item_emb(pos)).sum(dim=-1)
        neg_score = (user_emb * model.item_emb(neg)).sum(dim=-1)
        loss      = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    tqdm.write(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

print("학습 완료")

# ── 저장 ─────────────────────────────────────────────────
with open('./sasrec_model.pkl', 'wb') as f:
    pickle.dump({
        'model':        model.cpu(),
        'user_seq_idx': user_seq_idx,
        'user_target':  user_target,
        'iid_map':      iid_map,
        'idx2item':     idx2item,
        'uid_map':      uid_map,
    }, f)
print("저장 완료: ./sasrec_model.pkl")