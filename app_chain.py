import sys
sys.path.append("./Qwen3-VL-Embedding-2B")
sys.path.append("./Qwen3-VL-Reranker-2B")

import os
import torch
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
ALPHA              = 0.3
BETA               = 0.3
N_CANDIDATES       = 10
N_RESULTS          = 5
OWM_API_KEY        = "YOUR_OPENWEATHERMAP_API_KEY"

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

# ── LangChain 툴 정의 ─────────────────────────────────────
@tool
def get_weather_season(city: str) -> str:
    """도시 이름을 받아 현재 기온과 추천 패션 시즌을 반환한다. 도시, 날씨, 계절 언급이 있을 때 호출한다."""
    res  = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": OWM_API_KEY, "units": "metric"}
    )
    temp = res.json()["main"]["temp"]
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
    return f"추천 스타일: smart casual, versatile"

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

tools = [get_weather_season, get_fashion_trend, get_occasion_style]
agent          = create_tool_calling_agent(agent_llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

def enrich_query(natural_query: str) -> str:
    result   = agent_executor.invoke({"input": natural_query})
    enriched = result["output"]
    print(f"보강된 쿼리: {enriched}")
    return enriched

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
            emb_arr  = np.array([c[1] for c in candidates])
            bpr_arr  = np.array([c[2] for c in candidates])
            scores   = ALPHA * normalize(emb_arr) + (1 - ALPHA) * normalize(bpr_arr)
            ranked   = np.argsort(scores)[::-1]
            norm_emb = normalize(emb_arr)
            norm_bpr = normalize(bpr_arr)
            for i in ranked[:N_CANDIDATES]:
                print(f"{candidates[i][0]['prod_name'][:20]:20s} | emb={candidates[i][1]:.4f} bpr={candidates[i][2]:.4f} | norm_emb={norm_emb[i]:.4f} norm_bpr={norm_bpr[i]:.4f} | final={scores[i]:.4f}")
            return [candidates[i] for i in ranked[:N_CANDIDATES]]

    filtered = [(m, 1 - d, 0.0) for m, d in zip(metadatas, distances)
                if m['article_id'] not in purchased][:N_CANDIDATES]
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
    natural_query: Optional[str] = Form(None),
    use_rerank: bool = Form(False),
    image: Optional[UploadFile] = File(None)
):
    image_bytes = await image.read() if image else None

    enriched_query = None
    if natural_query:
        enriched_query = enrich_query(natural_query)

    candidates = get_candidates(enriched_query, image_bytes, customer_id)

    if use_rerank and candidates:
        final_metadatas, score_info = rerank(enriched_query, image_bytes, candidates)
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
            "article_id":     aid,
            "prod_name":      meta.get('prod_name', ''),
            "product_type":   info.get('product_type_name', ''),
            "colour":         info.get('colour_group_name', ''),
            "image_exists":   os.path.exists(get_image_path(aid)),
            "scores":         scores,
            "enriched_query": enriched_query,
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