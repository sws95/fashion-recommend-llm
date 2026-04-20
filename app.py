import sys
sys.path.append("./Qwen3-VL-Embedding-2B")
import os
import re
import json
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

app = FastAPI(title="Fashion Recommend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INSTRUCTION = "Retrieve fashion items relevant to the query."
MAX_PIXELS  = 256 * 256
ALPHA       = 0.3

# ── 모델 로딩 ─────────────────────────────────────────────
print("모델 로딩 중...")

embedding_model = Qwen3VLEmbedder(
    model_name_or_path="./Qwen3-VL-Embedding-2B",
    dtype=torch.bfloat16,
    max_pixels=MAX_PIXELS,
)

instruct_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./Qwen3-VL-2B-Instruct",
    dtype=torch.bfloat16,
    device_map="cuda"
)
instruct_processor = AutoProcessor.from_pretrained("./Qwen3-VL-2B-Instruct")

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("fashion_items")

with open('./bpr_model.pkl', 'rb') as f:
    bpr_data = pickle.load(f)

bpr_model         = bpr_data['model']
bpr_dataset       = bpr_data['dataset']
idx2item          = bpr_data['idx2item']
train_interaction = bpr_data['train_interaction']
train_user_items  = train_interaction.groupby('customer_id')['article_id'].apply(set).to_dict()

articles     = pd.read_csv('./articles.csv', dtype={'article_id': str})
article_info = articles.set_index('article_id').to_dict('index')

print("모델 로딩 완료")

# ── 유틸 ──────────────────────────────────────────────────
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

# ── 후보 추출 ─────────────────────────────────────────────
def get_candidates(query, image_bytes, customer_id, n_retrieve=50):
    inp = {"instruction": INSTRUCTION}
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
            emb_arr      = np.array([c[1] for c in candidates])
            bpr_arr      = np.array([c[2] for c in candidates])
            emb_norm_arr = normalize(emb_arr)
            bpr_norm_arr = normalize(bpr_arr)
            scores       = ALPHA * emb_norm_arr + (1 - ALPHA) * bpr_norm_arr
            ranked       = np.argsort(scores)[::-1]

            top = [(candidates[i][0], 1 - candidates[i][1], emb_norm_arr[i], bpr_norm_arr[i])
                   for i in ranked[:5]]
            return [t[0] for t in top], [t[1] for t in top], [t[2] for t in top], [t[3] for t in top]

    filtered  = [(m, d) for m, d in zip(metadatas, distances) if m['article_id'] not in purchased][:5]
    metadatas = [f[0] for f in filtered]
    distances = [f[1] for f in filtered]
    return metadatas, distances, [0.5] * len(metadatas), [0.0] * len(metadatas)

# ── 이유 생성 ─────────────────────────────────────────────
def generate_reason(meta, emb_norm, bpr_norm, query):
    colour       = meta.get('colour', '')
    product_type = meta.get('product_type', '')
    ratio        = bpr_norm / (emb_norm + 1e-9)

    if ratio > 2.0:
        return f"구매 히스토리 기반 추천 · {colour} {product_type}"
    elif ratio > 1.0:
        return f"취향과 검색어 모두 반영 · {colour} {product_type}"
    else:
        return f"'{query}' 검색어 매칭 · {colour} {product_type}"

# ── JSON 파싱 ─────────────────────────────────────────────
def parse_ranking(response, n):
    response = re.sub(r'```json|```', '', response).strip()
    match    = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            parsed  = json.loads(match.group())
            ranking = parsed.get('ranking', [])
            ranking = [int(r) for r in ranking if str(r).isdigit()]
            ranking = [r for r in ranking if 1 <= r <= n][:3]
            if ranking:
                return ranking
        except:
            pass
    print(f"파싱 실패, 원본: {response[:200]}")
    return list(range(1, min(4, n + 1)))

# ── LLM rerank ────────────────────────────────────────────
def llm_rerank(query, image_bytes, metadatas, distances):
    print(f"LLM rerank 시작, 후보 {len(metadatas)}개")
    content = []

    if image_bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": "위 이미지는 참고 이미지입니다.\n\n"})

    for i, meta in enumerate(metadatas, 1):
        img_path = get_image_path(meta['article_id'])
        if os.path.exists(img_path):
            content.append({"type": "image", "image": load_and_resize(img_path)})
        content.append({"type": "text", "text": f"[{i}]\n"})

    n           = len(metadatas)
    json_format = '{"ranking": [2, 3, 1]}'
    prompt = (
        f"아래 {n}개의 패션 아이템 이미지를 보고 검색어에 가장 잘 맞는 3개를 선택하세요.\n\n"
        f"규칙:\n"
        f"- JSON 형식만 출력. 다른 텍스트 금지.\n"
        f"- ranking: 1~{n} 사이 정수 3개 (잘 맞는 순서)\n"
        f"출력 형식:\n{json_format}\n\n"
    )
    if query:
        prompt += f"검색어: '{query}'\n검색어와 색상, 스타일, 종류가 가장 잘 맞는 아이템 3개를 고르세요."
    else:
        prompt += "참고 이미지와 색상, 실루엣, 소재가 가장 유사한 아이템 3개를 고르세요."

    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    inputs   = instruct_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output = instruct_model.generate(
            **inputs, max_new_tokens=64, repetition_penalty=1.2
        )

    trimmed  = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
    response = instruct_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"LLM 원본 응답: {response}")

    return parse_ranking(response, n)

# ── API 엔드포인트 ─────────────────────────────────────────
@app.post("/recommend")
async def recommend(
    customer_id: str = Form(...),
    query: Optional[str] = Form(None),
    use_llm: str = Form("false"),
    image: Optional[UploadFile] = File(None)
):
    use_llm_bool = use_llm.lower() == "true"
    image_bytes  = await image.read() if image else None

    metadatas, distances, emb_norms, bpr_norms = get_candidates(
        query, image_bytes, customer_id
    )

    if use_llm_bool and metadatas:
        ranking         = llm_rerank(query, image_bytes, metadatas, distances)
        final_metadatas = [metadatas[i-1] for i in ranking]
        final_emb_norms = [emb_norms[i-1] for i in ranking]
        final_bpr_norms = [bpr_norms[i-1] for i in ranking]
    else:
        final_metadatas = metadatas
        final_emb_norms = emb_norms
        final_bpr_norms = bpr_norms

    results = []
    for meta, emb_n, bpr_n in zip(final_metadatas, final_emb_norms, final_bpr_norms):
        aid    = meta['article_id']
        info   = article_info.get(aid, {})
        reason = generate_reason(meta, emb_n, bpr_n, query or "")
        print(f"  {aid} emb={emb_n:.3f} bpr={bpr_n:.3f} ratio={bpr_n/(emb_n+1e-9):.2f} → {reason}")
        results.append({
            "article_id":   aid,
            "prod_name":    meta.get('prod_name', ''),
            "product_type": info.get('product_type_name', ''),
            "colour":       info.get('colour_group_name', ''),
            "image_exists": os.path.exists(get_image_path(aid)),
            "reason":       reason
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