import sys
sys.path.append("./Qwen3-VL-Embedding-2B")
sys.path.append("./Qwen3-VL-Reranker-2B")

import os
import torch
import chromadb
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker import Qwen3VLReranker

app = FastAPI(title="Fashion Recommend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 상수 ──────────────────────────────────────────────────
EMBED_INSTRUCTION  = "Retrieve fashion items relevant to the query."
RERANK_INSTRUCTION = "Retrieve fashion items relevant to the user's query."
MAX_PIXELS         = 256 * 256
ALPHA              = 0.3   # embedding vs BPR
BETA               = 0.3   # reranker vs BPR
N_CANDIDATES       = 10
N_RESULTS          = 5

# ── 모델 로딩 ─────────────────────────────────────────────
print("모델 로딩 중...")

embedding_model = Qwen3VLEmbedder(
    model_name_or_path="./Qwen3-VL-Embedding-2B",
    torch_dtype=torch.bfloat16,
    max_pixels=MAX_PIXELS,
)

reranker_model = Qwen3VLReranker(
    model_name_or_path="./Qwen3-VL-Reranker-2B",
    torch_dtype=torch.bfloat16,
    max_pixels=MAX_PIXELS,
)

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("fashion_items")

with open('./bpr_model.pkl', 'rb') as f:
    bpr_data = pickle.load(f)

bpr_model         = bpr_data['model']
bpr_dataset       = bpr_data['dataset']
train_interaction = bpr_data['train_interaction']
train_user_items  = train_interaction.groupby('customer_id')['article_id'].apply(set).to_dict()

articles     = pd.read_csv('./articles.csv', dtype={'article_id': str})
article_info = articles.set_index('article_id').to_dict('index')

print("모델 로딩 완료")

# ── 유틸 함수 ─────────────────────────────────────────────
def get_image_path(article_id):
    folder = str(article_id)[:3]
    return f"./images/{folder}/{article_id}.jpg"

def load_and_resize(img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = (MAX_PIXELS / (w * h)) ** 0.5
    return img.resize((int(w * scale), int(h * scale)))

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

# ── 후보 검색 ─────────────────────────────────────────────
# 반환: List of (meta, emb_score, bpr_score)
def get_candidates(query, image_bytes, customer_id, n_retrieve=50):
    inp = {"instruction": EMBED_INSTRUCTION}
    if query:
        inp["text"] = query
    if image_bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        inp["image"] = img

    query_emb = embedding_model.process([inp])
    results   = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=n_retrieve
    )
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    purchased = train_user_items.get(customer_id, set())

    if customer_id in bpr_dataset.uid_map:
        user_idx    = bpr_dataset.uid_map[customer_id]
        item_scores = bpr_model.score(user_idx)

        candidates = []
        for meta, dist in zip(metadatas, distances):
            aid = meta['article_id']
            if aid in purchased:
                continue
            emb_score = 1 - dist
            bpr_score = float(item_scores[bpr_dataset.iid_map[aid]]) \
                        if aid in bpr_dataset.iid_map else 0.0
            candidates.append((meta, emb_score, bpr_score))

        if candidates:
            emb_arr = np.array([c[1] for c in candidates])
            bpr_arr = np.array([c[2] for c in candidates])
            scores  = ALPHA * normalize(emb_arr) + (1 - ALPHA) * normalize(bpr_arr)
            ranked  = np.argsort(scores)[::-1]
            #####
            norm_emb = normalize(emb_arr)
            norm_bpr = normalize(bpr_arr)
            for i in ranked[:N_CANDIDATES]:
                print(f"{candidates[i][0]['prod_name'][:20]:20s} | emb={candidates[i][1]:.4f} bpr={candidates[i][2]:.4f} | norm_emb={norm_emb[i]:.4f} norm_bpr={norm_bpr[i]:.4f} | final={scores[i]:.4f}")

            #####
            
            
            
            return [candidates[i] for i in ranked[:N_CANDIDATES]]

    # BPR 없으면 embedding만, bpr_score=0
    filtered = [(m, 1 - d, 0.0) for m, d in zip(metadatas, distances)
                if m['article_id'] not in purchased][:N_CANDIDATES]
    return filtered

# ── Reranker ──────────────────────────────────────────────
# 반환: (List[meta], List[score_info])
def rerank(query, image_bytes, candidates):
    metadatas  = [c[0] for c in candidates]
    bpr_scores = [c[2] for c in candidates]

    print(f"Reranker 시작, 후보 {len(metadatas)}개")

    query_input = {}
    if query:
        query_input["text"] = query
    if image_bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        query_input["image"] = img.resize((int(w * scale), int(h * scale)))

    documents = []
    for meta in metadatas:
        img_path = get_image_path(meta['article_id'])
        doc = {"text": meta['prod_name']}
        if os.path.exists(img_path):
            doc["image"] = load_and_resize(img_path)
        documents.append(doc)

    inputs = {
        "instruction": RERANK_INSTRUCTION,
        "query": query_input,
        "documents": documents,
        "fps": 1.0
    }

    rerank_scores = reranker_model.process(inputs)
    print(f"Reranker scores: {rerank_scores}")

    rerank_arr = np.array(rerank_scores)
    bpr_arr    = np.array(bpr_scores)
    final      = BETA * normalize(rerank_arr) + (1 - BETA) * normalize(bpr_arr)

    ranked = sorted(range(len(final)), key=lambda i: final[i], reverse=True)[:N_RESULTS]

    result_metadatas = [metadatas[i] for i in ranked]
    score_info = [{
        "rerank_score": round(float(rerank_scores[i]), 4),
        "bpr_score":    round(float(bpr_scores[i]), 4),
    } for i in ranked]

    return result_metadatas, score_info

# ── API 엔드포인트 ─────────────────────────────────────────
@app.post("/recommend")
async def recommend(
    customer_id: str = Form(...),
    query: Optional[str] = Form(None),
    use_rerank: bool = Form(False),
    image: Optional[UploadFile] = File(None)
):
    image_bytes = await image.read() if image else None

    candidates = get_candidates(query, image_bytes, customer_id)

    if use_rerank and candidates:
        final_metadatas, score_info = rerank(query, image_bytes, candidates)
    else:
        final_candidates = candidates[:N_RESULTS]
        final_metadatas  = [c[0] for c in final_candidates]
        score_info = [{
            "emb_score": round(float(c[1]), 4),
            "bpr_score": round(float(c[2]), 4),
        } for c in final_candidates]

    results = []
    for meta, scores in zip(final_metadatas, score_info):
        aid  = meta['article_id']
        info = article_info.get(aid, {})
        results.append({
            "article_id":   aid,
            "prod_name":    meta.get('prod_name', ''),
            "product_type": info.get('product_type_name', ''),
            "colour":       info.get('colour_group_name', ''),
            "image_exists": os.path.exists(get_image_path(aid)),
            "scores":       scores,
        })

    return JSONResponse({"results": results})

@app.get("/image/{article_id}")
async def get_image(article_id: str):
    from fastapi.responses import FileResponse
    img_path = get_image_path(article_id)
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/jpeg")
    return JSONResponse({"error": "이미지 없음"}, status_code=404)

@app.get("/health")
async def health():
    return {"status": "ok"}