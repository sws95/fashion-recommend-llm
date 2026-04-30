import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import os
import torch
import chromadb
import numpy as np
import pandas as pd
import pickle
import mlflow
from tqdm import tqdm
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# ── 데이터 로딩 ───────────────────────────────────────────
print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str, 'article_id': str
})
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].quantile(0.8)

train_df          = transactions[transactions['t_dat'] < split_date]
train_interaction = train_df.groupby(['customer_id', 'article_id']).size().reset_index(name='count')
train_user_items  = train_interaction.groupby('customer_id')['article_id'].apply(set).to_dict()

# ── BPR 로딩 ─────────────────────────────────────────────
print("BPR 모델 로딩...")
with open('./bpr_model.pkl', 'rb') as f:
    bpr_data = pickle.load(f)
bpr_model   = bpr_data['model']
bpr_dataset = bpr_data['dataset']
idx2item    = bpr_data['idx2item']

# ── Embedding 모델 로딩 ───────────────────────────────────
print("Embedding 모델 로딩...")
embedding_model = Qwen3VLEmbedder(
    model_name_or_path="./Qwen3-VL-Embedding-2B",
    dtype=torch.bfloat16,
    max_pixels=256*256,
)
client      = chromadb.PersistentClient(path="./chroma_db")
collection  = client.get_collection("fashion_items")
INSTRUCTION = "Retrieve fashion items relevant to the query."
print("모델 로딩 완료")

# ── 평가 지표 함수 ────────────────────────────────────────
def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

def hit_at_k(recommended, relevant, k):
    return 1.0 if set(recommended[:k]) & relevant else 0.0

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(recommended, relevant, k):
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

# ── 유저 & target 고정 ────────────────────────────────────
valid_users = [
    uid for uid, items in train_user_items.items()
    if len(items) >= 2 and uid in bpr_dataset.uid_map
]
np.random.seed(42)
fixed_users = np.random.choice(valid_users, min(1000, len(valid_users)), replace=False)

np.random.seed(42)
user_targets = {
    uid: np.random.choice(list(train_user_items[uid]))
    for uid in fixed_users
}
print(f"고정 유저 수: {len(fixed_users)}")

# ── article 메타 로딩 ─────────────────────────────────────
articles_df  = pd.read_csv('./articles.csv', dtype={'article_id': str})
article_meta = articles_df.set_index('article_id')[
    ['product_type_name', 'graphical_appearance_name', 'colour_group_name']
].to_dict('index')

# ── 평가 함수 ─────────────────────────────────────────────
def evaluate_leave_one_out(fixed_users, user_targets, k=5, n_retrieve=50, alpha=0.7):
    bpr_hits,    emb_hits,    hybrid_hits    = [], [], []
    bpr_ndcgs,   emb_ndcgs,   hybrid_ndcgs   = [], [], []
    bpr_mrrs,    emb_mrrs,    hybrid_mrrs    = [], [], []

    for customer_id in tqdm(fixed_users, desc=f"alpha={alpha} k={k}"):
        target    = user_targets[customer_id]
        remaining = train_user_items[customer_id] - {target}

        if target not in article_meta:
            continue

        meta  = article_meta[target]
        query = f"{meta['product_type_name']} {meta['graphical_appearance_name']} {meta['colour_group_name']}"

        user_idx    = bpr_dataset.uid_map[customer_id]
        item_scores = bpr_model.score(user_idx)

        recommended_bpr = []
        for idx in np.argsort(item_scores)[::-1]:
            aid = idx2item[idx]
            if aid not in remaining:
                recommended_bpr.append(aid)
            if len(recommended_bpr) == k:
                break

        bpr_hits.append(hit_at_k(recommended_bpr, {target}, k))
        bpr_ndcgs.append(ndcg_at_k(recommended_bpr, {target}, k))
        bpr_mrrs.append(mrr_at_k(recommended_bpr, {target}, k))

        inp       = {"instruction": INSTRUCTION, "text": query}
        query_emb = embedding_model.process([inp])
        results   = collection.query(query_embeddings=query_emb.tolist(), n_results=n_retrieve)
        emb_metadatas = results['metadatas'][0]
        emb_distances = results['distances'][0]

        recommended_emb = []
        for emb_meta in emb_metadatas:
            aid = emb_meta['article_id']
            if aid not in remaining:
                recommended_emb.append(aid)
            if len(recommended_emb) == k:
                break

        emb_hits.append(hit_at_k(recommended_emb, {target}, k))
        emb_ndcgs.append(ndcg_at_k(recommended_emb, {target}, k))
        emb_mrrs.append(mrr_at_k(recommended_emb, {target}, k))

        candidates = []
        for emb_meta, dist in zip(emb_metadatas, emb_distances):
            aid = emb_meta['article_id']
            if aid in remaining:
                continue
            emb_score = 1 - dist
            bpr_score = float(item_scores[bpr_dataset.iid_map[aid]]) \
                        if aid in bpr_dataset.iid_map else 0.0
            candidates.append((aid, emb_score, bpr_score))

        if not candidates:
            hybrid_hits.append(0.0)
            hybrid_ndcgs.append(0.0)
            hybrid_mrrs.append(0.0)
            continue

        emb_arr      = np.array([c[1] for c in candidates])
        bpr_arr      = np.array([c[2] for c in candidates])
        final_scores = alpha * normalize(emb_arr) + (1 - alpha) * normalize(bpr_arr)
        ranked_idx   = np.argsort(final_scores)[::-1]
        recommended_hybrid = [candidates[i][0] for i in ranked_idx[:k]]

        hybrid_hits.append(hit_at_k(recommended_hybrid, {target}, k))
        hybrid_ndcgs.append(ndcg_at_k(recommended_hybrid, {target}, k))
        hybrid_mrrs.append(mrr_at_k(recommended_hybrid, {target}, k))

    return {
        "BPR":    {"HitRate": np.mean(bpr_hits),    "NDCG": np.mean(bpr_ndcgs),    "MRR": np.mean(bpr_mrrs)},
        "Emb":    {"HitRate": np.mean(emb_hits),    "NDCG": np.mean(emb_ndcgs),    "MRR": np.mean(emb_mrrs)},
        "Hybrid": {"HitRate": np.mean(hybrid_hits), "NDCG": np.mean(hybrid_ndcgs), "MRR": np.mean(hybrid_mrrs)},
    }

# ── 실행 ──────────────────────────────────────────────────
mlflow.set_experiment("fashion-recsys")

alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

with mlflow.start_run(run_name="base_model"):
    mlflow.log_artifact("./bpr_model.pkl")
    mlflow.log_param("embedding_model", "Qwen3-VL-Embedding-2B")

for k in [5, 10]:
    for alpha in alphas:
        with mlflow.start_run(run_name=f"hybrid_k{k}_a{alpha}"):
            mlflow.log_param("k", k)
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("n_retrieve", 50)
            mlflow.log_param("embedding_model", "Qwen3-VL-Embedding-2B")

            scores = evaluate_leave_one_out(fixed_users, user_targets, k=k, alpha=alpha)

            for model_name, metrics in scores.items():
                prefix = model_name.lower()
                mlflow.log_metric(f"{prefix}_hitrate", metrics['HitRate'])
                mlflow.log_metric(f"{prefix}_ndcg",    metrics['NDCG'])
                mlflow.log_metric(f"{prefix}_mrr",     metrics['MRR'])

            print(f"k={k} alpha={alpha}")
            for model_name, metrics in scores.items():
                print(f"  {model_name:<10} HitRate={metrics['HitRate']:.4f} NDCG={metrics['NDCG']:.4f} MRR={metrics['MRR']:.4f}")