import sys
sys.path.append("./Qwen3-VL-Embedding-2B")
sys.path.append("./Qwen3-VL-Reranker-2B")

import os
import time
import torch
import torch.nn as nn
import chromadb
import numpy as np
import pandas as pd
import pickle
import requests
from PIL import Image
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker import Qwen3VLReranker
from sasrec import SASRec  # 클래스 정의 대신 import
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
import json

with open('./config.json', 'r') as f:
    config = json.load(f)

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
MAX_PIXELS_RERANK  = 128 * 128
ALPHA              = 0.3
BETA               = 0.3
N_CANDIDATES       = 5
N_RESULTS          = 5
N_RELATED          = 5
MAX_SEQ_LEN        = 50
POP_ALPHA          = 0.01
EPSILON            = 0.1
OWM_API_KEY        = config["OWM_API_KEY"]

# ── SASRec 모델 정의 ──────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim, n_heads, n_layers, max_len, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, embed_dim)
        self.dropout  = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, seq):
        batch_size, seq_len = seq.shape
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x   = self.item_emb(seq) + self.pos_emb(pos)
        x   = self.dropout(x)
        pad_mask    = (seq == 0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device), diagonal=1
        ).bool()
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        last_idx = (seq != 0).sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        return x[torch.arange(batch_size), last_idx]

    def predict(self, seq, item_indices):
        user_emb = self.forward(seq)
        item_emb = self.item_emb(item_indices)
        return (user_emb.unsqueeze(1) * item_emb).sum(-1)

# ── 모델 로딩 ─────────────────────────────────────────────
print("모델 로딩 중...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
item_popularity   = train_interaction.groupby('article_id')['customer_id'].count().to_dict()

articles     = pd.read_csv('./articles.csv', dtype={'article_id': str})
article_info = articles.set_index('article_id').to_dict('index')

print("SASRec 모델 로딩...")
with open('./sasrec_model.pkl', 'rb') as f:
    sasrec_data = pickle.load(f)

sasrec_model    = sasrec_data['model'].to(device)
sasrec_model.eval()
sasrec_uid_map  = sasrec_data['uid_map']
sasrec_iid_map  = sasrec_data['iid_map']
sasrec_idx2item = sasrec_data['idx2item']
sasrec_seq_idx  = sasrec_data['user_seq_idx']

sasrec_idx2uid = {v: k for k, v in sasrec_uid_map.items()}
sasrec_seq_str = {
    sasrec_idx2uid[idx]: seq
    for idx, seq in sasrec_seq_idx.items()
    if idx in sasrec_idx2uid
}

n_sasrec_items  = len(sasrec_iid_map) + 1
all_item_tensor = torch.arange(1, n_sasrec_items, dtype=torch.long).to(device).unsqueeze(0)

print("모델 로딩 완료")

# ── LangChain 툴 정의 ─────────────────────────────────────
@tool
def get_weather_season(city: str) -> str:
    """도시 이름을 영어로 받아 현재 기온과 추천 패션 시즌을 반환한다.
    city는 반드시 영어로 변환해서 전달할 것. 예: 서울→Seoul, 부산→Busan"""
    res  = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": OWM_API_KEY, "units": "metric"}
    )
    data = res.json()
    print(data)
    if "main" not in data:
        raise ValueError(f"날씨 API 오류: {data}")
    temp = data["main"]["temp"]
    if temp >= 23:
        season = "summer"
    elif temp >= 15:
        season = "spring"
    elif temp >= 5:
        season = "autumn"
    else:
        season = "winter"
    return f"현재 기온: {temp}°C, 추천 시즌: {season}"

@tool
def get_fashion_trend(season: str) -> str:
    """시즌을 받아 현재 패션 트렌드 키워드를 검색한다. '요즘 유행', '트렌드' 언급이 있을 때 호출한다."""
    search  = TavilySearchResults(max_results=3)
    results = search.invoke(f"{season} 2026 fashion trend keywords")
    return str(results)

@tool
def get_occasion_style(occasion: str) -> str:
    """일정이나 장소를 받아 적합한 패션 스타일을 반환한다. 미팅, 데이트, 여행 등 일정/장소 언급이 있을 때 호출한다."""
    style_map = {
        "미팅": "smart casual, business casual, neat collared shirt, slacks",
        "meeting": "smart casual, business casual, neat collared shirt, slacks",
        "비즈니스": "formal, suit, dress shirt, blazer",
        "business": "formal, suit, dress shirt, blazer",
        "데이트": "smart casual, romantic, stylish, clean fit",
        "date": "smart casual, romantic, stylish, clean fit",
        "여행": "casual, comfortable, functional, light layers",
        "travel": "casual, comfortable, functional, light layers",
        "야외": "casual, outdoor, comfortable, layered",
        "outdoor": "casual, outdoor, comfortable, layered",
        "파티": "party, dressy, bold, statement piece",
        "party": "party, dressy, bold, statement piece",
        "운동": "sportswear, athletic, activewear",
        "workout": "sportswear, athletic, activewear",
    }
    for key, style in style_map.items():
        if key in occasion.lower():
            return f"추천 스타일: {style}"
    return "추천 스타일: smart casual, versatile"

# ── 에이전트 구성 ─────────────────────────────────────────
agent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=config["GEMINI_API_KEY"]
)

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 오늘의 옷을 추천하기 위한 쿼리 보강 에이전트입니다.
유저 입력을 분석해서 필요한 툴만 골라 호출하세요.

- 도시/날씨/계절 언급 있으면 → get_weather_season 호출
- '요즘 유행', '트렌드' 언급 있으면 → get_fashion_trend 호출
- 미팅/데이트/여행 등 일정이나 장소 언급 있으면 → get_occasion_style 호출
- 해당되는 툴 모두 호출 가능

툴 결과를 종합해서 패션 아이템 검색에 적합한 영어 키워드 쿼리를 한 문장으로 만들어 출력하세요.
툴이 필요 없으면 원래 입력을 영어로 번역해서 반환하세요.
반드시 최종 쿼리만 출력하세요."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools          = [get_weather_season, get_fashion_trend, get_occasion_style]
agent          = create_tool_calling_agent(agent_llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

def enrich_query(natural_query: str) -> str:
    t = time.time()
    result   = agent_executor.invoke({"input": natural_query})
    enriched = result["output"]
    print(f"[시간] 쿼리 보강: {time.time()-t:.2f}초")
    print(f"보강된 쿼리: {enriched}")
    return enriched

# ── 유틸 함수 ─────────────────────────────────────────────
def get_image_path(article_id):
    folder = str(article_id)[:3]
    return f"./images/{folder}/{article_id}.jpg"

def load_and_resize(img_path, max_pixels=MAX_PIXELS):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((int(w * scale), int(h * scale)))

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

# ── 후보 검색 ─────────────────────────────────────────────
def get_candidates(query, image_bytes, customer_id, n_retrieve=50):
    t = time.time()

    inp = {"instruction": EMBED_INSTRUCTION}
    if query:
        inp["text"] = query
    if image_bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        inp["image"] = img

    query_emb = embedding_model.process([inp])
    print(f"[시간] 쿼리 임베딩: {time.time()-t:.2f}초")

    t2 = time.time()
    results   = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=n_retrieve
    )
    print(f"[시간] ChromaDB 검색: {time.time()-t2:.2f}초")

    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    purchased = train_user_items.get(customer_id, set())

    if customer_id in bpr_dataset.uid_map:
        t3 = time.time()
        user_idx    = bpr_dataset.uid_map[customer_id]
        item_scores = bpr_model.score(user_idx)

        candidates = []
        for meta, dist in zip(metadatas, distances):
            aid = meta['article_id']
            if aid in purchased:
                continue
            emb_score = 1 - dist
            # BPR 없는 아이템은 min값으로
            bpr_score = float(item_scores[bpr_dataset.iid_map[aid]]) \
                        if aid in bpr_dataset.iid_map else float(item_scores.min())
            candidates.append((meta, emb_score, bpr_score))

        if candidates:
            emb_arr  = np.array([c[1] for c in candidates])
            bpr_arr  = np.array([c[2] for c in candidates])

            # 순수 hybrid score
            scores = ALPHA * normalize(emb_arr) + (1 - ALPHA) * normalize(bpr_arr)
            ranked = list(np.argsort(scores)[::-1])

            # 4등: 확률적으로 popularity 패널티 적용 1등
            if np.random.random() < EPSILON:
                pop_arr     = np.array([item_popularity.get(c[0]['article_id'], 1) for c in candidates])
                pop_scores  = scores / (pop_arr ** POP_ALPHA)
                pop_top_idx = np.argmax(pop_scores)
                if pop_top_idx not in ranked[:3]:
                    ranked[3] = pop_top_idx

            # 5등: 확률적으로 emb 높고 bpr 낮은 아이템
            if np.random.random() < EPSILON:
                explore_score = normalize(emb_arr) - normalize(bpr_arr)
                explore_idx   = np.argmax(explore_score)
                if explore_idx not in ranked[:4]:
                    ranked[4] = explore_idx

            norm_emb = normalize(emb_arr)
            norm_bpr = normalize(bpr_arr)
            for i in ranked[:N_CANDIDATES]:
                print(f"{candidates[i][0]['prod_name'][:20]:20s} | emb={candidates[i][1]:.4f} bpr={candidates[i][2]:.4f} | norm_emb={norm_emb[i]:.4f} norm_bpr={norm_bpr[i]:.4f} | final={scores[i]:.4f}")
            print(f"[시간] BPR 스코어링: {time.time()-t3:.2f}초")
            print(f"[시간] 후보 검색 전체: {time.time()-t:.2f}초")
            return [candidates[i] for i in ranked[:N_CANDIDATES]]

    filtered = [(m, 1 - d, float(item_scores.min()) if customer_id in bpr_dataset.uid_map else 0.0)
                for m, d in zip(metadatas, distances)
                if m['article_id'] not in purchased][:N_CANDIDATES]
    print(f"[시간] 후보 검색 전체: {time.time()-t:.2f}초")
    return filtered

# ── Reranker ──────────────────────────────────────────────
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

    t = time.time()
    documents = []
    for meta in metadatas:
        img_path = get_image_path(meta['article_id'])
        doc = {"text": meta['prod_name']}
        if os.path.exists(img_path):
            doc["image"] = load_and_resize(img_path, max_pixels=MAX_PIXELS_RERANK)
        documents.append(doc)
    print(f"[시간] 이미지 로딩: {time.time()-t:.2f}초")

    inputs = {
        "instruction": RERANK_INSTRUCTION,
        "query": query_input,
        "documents": documents,
        "fps": 1.0
    }

    t2 = time.time()
    rerank_scores = reranker_model.process(inputs)
    print(f"[시간] reranker 추론: {time.time()-t2:.2f}초")
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
    natural_query: Optional[str] = Form(None),
    use_rerank: bool = Form(False),
    image: Optional[UploadFile] = File(None)
):
    t_total = time.time()
    image_bytes = await image.read() if image else None

    enriched_query = None
    if natural_query:
        enriched_query = enrich_query(natural_query)

    candidates = get_candidates(enriched_query, image_bytes, customer_id)

    if use_rerank and candidates:
        t = time.time()
        final_metadatas, score_info = rerank(enriched_query, image_bytes, candidates)
        print(f"[시간] rerank 전체: {time.time()-t:.2f}초")
    else:
        final_metadatas = [c[0] for c in candidates[:N_RESULTS]]
        score_info = [{
            "emb_score": round(float(c[1]), 4),
            "bpr_score": round(float(c[2]), 4),
        } for c in candidates[:N_RESULTS]]

    results = []
    for idx, (meta, scores) in enumerate(zip(final_metadatas, score_info)):
        aid  = meta['article_id']
        info = article_info.get(aid, {})
        results.append({
            "article_id":     aid,
            "prod_name":      meta.get('prod_name', ''),
            "product_type":   info.get('product_type_name', ''),
            "colour":         info.get('colour_group_name', ''),
            "description":    info.get('detail_desc', '') if idx == 0 else '',
            "image_exists":   os.path.exists(get_image_path(aid)),
            "scores":         scores,
            "enriched_query": enriched_query,
        })

    print(f"[시간] 전체 요청: {time.time()-t_total:.2f}초")
    return JSONResponse({"results": results})

# ── SASRec 연관 추천 엔드포인트 ──────────────────────────
@app.post("/recommend/related")
async def recommend_related(customer_id: str = Form(...)):
    if customer_id not in sasrec_seq_str:
        return JSONResponse({"results": [], "message": "구매 이력 없음"})

    seq     = sasrec_seq_str[customer_id]
    seq     = seq[-MAX_SEQ_LEN:]
    pad_len = MAX_SEQ_LEN - len(seq)
    seq     = [0] * pad_len + seq

    seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)

    t = time.time()
    with torch.no_grad():
        scores = sasrec_model.predict(seq_tensor, all_item_tensor)[0].cpu().numpy()
    print(f"[시간] SASRec 추론: {time.time()-t:.2f}초")

    purchased   = train_user_items.get(customer_id, set())
    top_indices = scores.argsort()[::-1]

    results = []
    for idx in top_indices:
        aid = sasrec_idx2item.get(idx + 1)
        if aid is None or aid in purchased:
            continue
        info = article_info.get(aid, {})
        results.append({
            "article_id":   aid,
            "prod_name":    info.get('prod_name', ''),
            "product_type": info.get('product_type_name', ''),
            "colour":       info.get('colour_group_name', ''),
            "image_exists": os.path.exists(get_image_path(aid)),
        })
        if len(results) == N_RELATED:
            break

    print(f"[시간] SASRec 연관 추천 전체: {time.time()-t:.2f}초")
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