import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import json
import time
import pickle
import requests
import torch
import numpy as np
import pandas as pd

# /recommend 전체 latency (온라인 날씨 API 포함, Tavily·Gemini 제외)
#   한국어 요청 → [로컬 Qwen3-VL-2B-Instruct 1회] 영어 쿼리 + 도시 추출 → (도시 있으면) 날씨 API → 계절 키워드 추가
#   → 쿼리 임베딩 → 검색 top1000 ∪ Two-Tower top50 → Two-Tower 가중합 → 상위 5
# 날씨: app의 OpenWeatherMap은 키(config.json)가 없어 Open-Meteo(무키)로 대체 — 지오코딩 + 현재 기온 2회 호출
# 8GB GPU라 Instruct/Embedding 동시 적재 불가 → 단계 A(보강+날씨), 단계 B(임베딩~정렬)를 요청별로 따로 재서 합산
# Reranker(3.2초, README)는 모델 부재로 별도 표기

REQUESTS = [
    "서울 날씨에 맞는 출근룩 추천해줘", "부산 여행 갈 때 입을 옷 추천", "오늘 제주도 날씨에 입을 원피스",
    "대구에서 주말 데이트할 때 입을 옷", "인천 공항 갈 때 편한 옷", "운동할 때 입기 좋은 레깅스",
    "파티에 입을 화려한 드레스", "광주 날씨 보고 니트 추천해줘", "캐주얼한 청바지 추천", "대전 출장 갈 때 셔츠",
]
N_ROUNDS = 2                                     # 요청 10개 × 2회 = 20건 (+ warm-up 1건)
device = torch.device("cuda")
sync = torch.cuda.synchronize

def season_of(temp):                             # app_chain.py get_weather_season과 동일 기준
    return "summer" if temp >= 23 else "spring" if temp >= 15 else "autumn" if temp >= 5 else "winter"

# ── 단계 A: 쿼리 보강 (로컬 LLM) + 날씨 API ──────────────
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
proc = AutoProcessor.from_pretrained("./Qwen3-VL-2B-Instruct")
llm  = Qwen3VLForConditionalGeneration.from_pretrained("./Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()
SYSTEM = ('Convert the Korean fashion request to JSON: {"query": short English keyword query for product search, '
          '"city": English city name if a Korean city is mentioned else null}. Output only JSON.')
http = requests.Session()                        # keep-alive (서비스처럼 연결 재사용)

def enrich(text):
    t = {}
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": text}]}]
    ids = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
    sync(); s = time.perf_counter()
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=48, do_sample=False)
    sync(); t['로컬 LLM (번역 + 도시 추출)'] = (time.perf_counter() - s) * 1000
    raw = proc.batch_decode(out[:, ids['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    try:
        parsed = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
    except ValueError:
        parsed = {"query": raw.strip(), "city": None}
    query, city = parsed.get("query") or text, parsed.get("city")
    t['날씨 API (지오코딩 + 기온)'] = 0.0
    if city:
        s = time.perf_counter()
        g = http.get('https://geocoding-api.open-meteo.com/v1/search', params={'name': city, 'count': 1}, timeout=10).json()
        if g.get('results'):
            g = g['results'][0]
            w = http.get('https://api.open-meteo.com/v1/forecast',
                         params={'latitude': g['latitude'], 'longitude': g['longitude'], 'current': 'temperature_2m'}, timeout=10).json()
            query = f"{query} {season_of(w['current']['temperature_2m'])}"
        t['날씨 API (지오코딩 + 기온)'] = (time.perf_counter() - s) * 1000
    return query, city, t

stage_a, examples = [], []
enrich(REQUESTS[0])                              # warm-up (GPU + HTTP 연결)
for r in range(N_ROUNDS):
    for text in REQUESTS:
        q, city, t = enrich(text)
        stage_a.append(t)
        if r == 0:
            examples.append((text, city, q))
print("보강 예시:")
for e in examples:
    print("  ", e)
del llm
torch.cuda.empty_cache()

# ── 단계 B: 임베딩 → 검색 → Two-Tower → 정렬 ─────────────
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
emb_model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16)
INSTRUCTION = "Retrieve fashion items relevant to the query."
item_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
item_vec = torch.tensor(np.load('./qwen_text_emb.npy'), device=device)
with open('./twotower_qwen_model.pkl', 'rb') as f:
    d = pickle.load(f)
tt_U = torch.tensor(d['user_emb'], device=device)
tt_I = torch.tensor(d['item_emb'], device=device)
users = np.random.RandomState(0).choice(len(d['user_ids']), len(stage_a) + 1, replace=False)
del d

def normalize(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

def recommend(q, u):
    t = {}
    sync(); s = time.perf_counter()
    q_vec = emb_model.process([{"text": q, "instruction": INSTRUCTION}]).to(device).half()[0]
    sync(); t['쿼리 임베딩'] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    qs = item_vec @ q_vec
    cf = tt_I @ tt_U[u]
    cand = torch.unique(torch.cat([qs.topk(1000).indices, cf.topk(50).indices]))
    e, c = qs[cand].float().cpu().numpy(), cf[cand].float().cpu().numpy()
    top5 = cand[torch.tensor(np.argsort(-(0.4 * normalize(e) + 0.6 * normalize(c)))[:5].copy())]
    sync(); t['검색 + Two-Tower + 정렬'] = (time.perf_counter() - s) * 1000
    return [item_ids[i] for i in top5.tolist()], t

recommend("warm-up", users[-1])
qs_all = [e[2] for e in examples] * N_ROUNDS
stage_b = [recommend(q, u)[1] for q, u in zip(qs_all, users)]

# ── 출력 ─────────────────────────────────────────────────
df = pd.concat([pd.DataFrame(stage_a), pd.DataFrame(stage_b)], axis=1)
df['합계 (rerank 없음)'] = df.sum(axis=1)
with_city = df['날씨 API (지오코딩 + 기온)'] > 0
print(f"\n=== 요청 {len(df)}건 (도시 언급 {with_city.sum()}건), ms ===")
print(f"{'단계':<28}{'평균':>9}{'p50':>9}{'p95':>9}")
for col in df.columns:
    v = df.loc[with_city, col] if col.startswith('날씨') else df[col]
    print(f"{col:<28}{v.mean():>9.1f}{v.median():>9.1f}{np.percentile(v, 95):>9.1f}")
for label, m in (('도시 언급 O', with_city), ('도시 언급 X', ~with_city)):
    print(f"합계 ({label}){'':<14}{df.loc[m, '합계 (rerank 없음)'].mean():>9.1f}")
print("+ Reranker (README 3.2초) 적용 시 합계 + 3,200ms")
