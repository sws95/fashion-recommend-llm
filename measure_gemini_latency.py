import os
import json
import time
import requests
import numpy as np
import pandas as pd

# Gemini 쿼리 보강 latency (app_chain.py enrich_query 대체 측정)
#   simple: 1회 호출로 영어 쿼리 + 도시 추출 (로컬 LLM과 같은 작업)
#   agent : app의 LangChain tool-calling agent와 같은 흐름 — LLM(도구 선택) → 날씨 API → LLM(최종 쿼리)
# app의 gemini-2.5-flash는 신규 사용자 미제공 → gemini-flash-latest / flash-lite-latest로 측정
# 날씨는 OpenWeatherMap 키가 없어 Open-Meteo 사용, Tavily 제외
# 키: 환경변수 GEMINI_API_KEY

KEY      = os.environ['GEMINI_API_KEY']
MODELS   = os.environ.get('GEMINI_MODELS', 'gemini-flash-latest,gemini-flash-lite-latest').split(',')
REQUESTS = [
    "서울 날씨에 맞는 출근룩 추천해줘", "부산 여행 갈 때 입을 옷 추천", "오늘 제주도 날씨에 입을 원피스",
    "대구에서 주말 데이트할 때 입을 옷", "인천 공항 갈 때 편한 옷", "운동할 때 입기 좋은 레깅스",
    "파티에 입을 화려한 드레스", "광주 날씨 보고 니트 추천해줘", "캐주얼한 청바지 추천", "대전 출장 갈 때 셔츠",
]
http = requests.Session()
PAUSE = float(os.environ.get('GEMINI_PAUSE', 6))      # 호출 간격 (짧은 키의 분당 한도 대응)
N_REQ = int(os.environ.get('GEMINI_N', 5))

def gemini(model, body):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    for attempt in range(4):                     # 429(분당 한도) → 대기 후 재시도, 대기 시간은 측정에서 제외
        time.sleep(PAUSE)
        s = time.perf_counter()
        r = http.post(url, json=body, headers={'x-goog-api-key': KEY}, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(20)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - s) * 1000

def weather(city):
    s = time.perf_counter()
    g = http.get('https://geocoding-api.open-meteo.com/v1/search', params={'name': city, 'count': 1}, timeout=10).json()
    if not g.get('results'):
        return "날씨 정보 없음", (time.perf_counter() - s) * 1000
    g = g['results'][0]
    temp = http.get('https://api.open-meteo.com/v1/forecast', params={'latitude': g['latitude'], 'longitude': g['longitude'],
                    'current': 'temperature_2m'}, timeout=10).json()['current']['temperature_2m']
    season = "summer" if temp >= 23 else "spring" if temp >= 15 else "autumn" if temp >= 5 else "winter"
    return f"현재 기온: {temp}°C, 추천 시즌: {season}", (time.perf_counter() - s) * 1000

def text_of(resp):
    return ''.join(p.get('text', '') for p in resp['candidates'][0]['content'].get('parts', []))

# ── simple ──
SIMPLE_PROMPT = ('Convert the Korean fashion request to JSON: {"query": short English keyword query for product search, '
                 '"city": English city name if a Korean city is mentioned else null}. Output only JSON.\n\nRequest: ')

def run_simple(model, text):
    t = {}
    resp, t['LLM 호출 1'] = gemini(model, {'contents': [{'role': 'user', 'parts': [{'text': SIMPLE_PROMPT + text}]}]})
    raw = text_of(resp)
    try:
        p = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
    except ValueError:
        p = {'query': raw.strip(), 'city': None}
    q = p.get('query') or text
    t['날씨 API'] = 0.0
    if p.get('city'):
        w, t['날씨 API'] = weather(p['city'])
        q = f"{q} {w.split('추천 시즌: ')[-1]}"
    return q, t

# ── agent (app_chain.py 프롬프트 + get_weather_season 도구) ──
AGENT_SYSTEM = """당신은 오늘의 옷을 추천하기 위한 쿼리 보강 에이전트입니다.
유저 입력을 분석해서 필요한 툴만 골라 호출하세요.
- 도시/날씨/계절 언급 있으면 → get_weather_season 호출
툴 결과를 종합해서 패션 아이템 검색에 적합한 영어 키워드 쿼리를 한 문장으로 만들어 출력하세요.
툴이 필요 없으면 원래 입력을 영어로 번역해서 반환하세요.
반드시 최종 쿼리만 출력하세요."""
TOOLS = [{'functionDeclarations': [{
    'name': 'get_weather_season',
    'description': '도시 이름을 영어로 받아 현재 기온과 추천 패션 시즌을 반환한다. city는 반드시 영어로 변환해서 전달할 것.',
    'parameters': {'type': 'OBJECT', 'properties': {'city': {'type': 'STRING'}}, 'required': ['city']}}]}]

def run_agent(model, text):
    t = {'날씨 API': 0.0}
    contents = [{'role': 'user', 'parts': [{'text': text}]}]
    body = lambda: {'systemInstruction': {'parts': [{'text': AGENT_SYSTEM}]}, 'tools': TOOLS, 'contents': contents}
    n = 0
    while True:
        n += 1
        resp, t[f'LLM 호출 {n}'] = gemini(model, body())
        content = resp['candidates'][0]['content']
        calls = [p['functionCall'] for p in content.get('parts', []) if 'functionCall' in p]
        if not calls or n >= 4:
            return text_of(resp).strip(), t
        contents.append(content)                 # thought signature 포함 그대로 전달
        parts = []
        for c in calls:
            w, ms = weather(c['args'].get('city', ''))
            t['날씨 API'] += ms
            parts.append({'functionResponse': {'name': c['name'], 'response': {'result': w}}})
        contents.append({'role': 'user', 'parts': parts})

rows, examples = [], []
for model in MODELS:
    for mode, fn in (('simple', run_simple), ('agent', run_agent)):
        try:
            fn(model, REQUESTS[0])               # warm-up (연결)
        except Exception as e:
            print(f"{model} {mode} 실패: {e}")
            continue
        for text in REQUESTS[:N_REQ]:
            q, t = fn(model, text)
            llm = sum(v for k, v in t.items() if k.startswith('LLM'))
            rows.append({'model': model, 'mode': mode, 'LLM 호출 수': sum(k.startswith('LLM') for k in t),
                         'LLM 시간': llm, '날씨 API': t['날씨 API'], '합계': llm + t['날씨 API']})
            examples.append((model, mode, text, q))

df = pd.DataFrame(rows)
print(f"\n=== 쿼리 보강 latency (요청 {N_REQ}건, ms) ===")
print(df.groupby(['model', 'mode']).agg(LLM호출수=('LLM 호출 수', 'mean'), LLM=('LLM 시간', 'mean'),
                                         날씨=('날씨 API', 'mean'), 합계=('합계', 'mean'),
                                         합계_p95=('합계', lambda v: np.percentile(v, 95))).round(1))
print("\n예시:")
for e in examples:
    print("  ", e)
