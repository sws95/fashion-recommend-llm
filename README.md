## Dataset
- H&M Personalized Fashion Recommendations (Kaggle)
- 105,542 items / 1,371,980 users / 31,788,324 transactions
- BPR 학습: 1,240,163 users / 91,798 items / 21,793,296 interactions (2018~2020 구매 데이터, 시간 기준 80% split)

## Tech Stack
- **Language**: Python, PyTorch
- **Embedding**: Qwen3-VL-Embedding-2B (multimodal, text + image)
- **Vector DB**: ChromaDB (105,542 items, 256×256)
- **Collaborative Filtering**: BPR (Bayesian Personalized Ranking), Two-Tower (+Qwen 콘텐츠 임베딩), SASRec (Sequential)
- **VL Reranker**: Qwen3-VL-Reranker-2B (local, 128×128 최적화) + article description 기반 추천 이유 제공
- **Query Agent**: LangChain + Gemini 2.5 Flash (날씨/트렌드/일정 툴 호출)

## Serving Latency
| 단계 | rerank 없음 | rerank 있음 |
|------|------------|------------|
| 쿼리 보강 (Gemini) | 5~7초 | 5~7초 |
| 쿼리 임베딩 | 0.5초 | 0.2초 |
| ChromaDB 검색 | 0.7초 | 0.1초 |
| BPR 스코어링 | 0.01초 | 0.01초 |
| reranker 추론 | - | 3.2초 |
| **전체** | **~8초** | **~11초** |

> Reranker 이미지 해상도 256×256 → 128×128 최적화로 추론 시간 16초 → 3.2초로 단축 (3.5배)  
> 쿼리 보강 Gemini API(5~7초) → Qwen3-VL-2B-Instruct 로컬 추론으로 대체 시 0.7초로 단축 가능.  
> VRAM 16GB 제약 (Embedding + Reranker 선점)으로 미적용. 추천 이유 생성은 LLM 생성 대신 article description으로 대체.

## Evaluation Results

### Hybrid vs Baseline (Leave-one-out, 1,000 users, 105,542 items)
| Model | HitRate@5 | HitRate@10 |
|-------|-----------|------------|
| BPR only | 0.005 | 0.010 |
| Embedding only | 0.079 | 0.121 |
| **Hybrid (alpha=0.3)** | **0.211** | **0.264** |

> Embedding only 대비 Hybrid: HitRate@5 기준 **+167%** 향상
> 평가 쿼리 : 정답 아이템의 product_type + graphical_appearance + colour_group
> ⚠️ 이 평가는 train 구매를 정답으로 써서 BPR이 정답을 학습 때 본 상태(leak). 미래 구매 기준 재평가는 [아래](#two-tower--재평가-미래-구매-기준) 참고.

### Embedding Ablation — CLIP vs Qwen3-VL (Leave-one-out, 1,000 users, 10,000 items)
| Model | HitRate@5 | HitRate@10 |
|-------|-----------|------------|
| CLIP (fashion-clip) | 0.331 | 0.426 |
| Qwen3-VL v1 (색깔 포함) | **0.489** | **0.600** |
| Qwen3-VL v2 (색깔 제외) | 0.388 | 0.477 |

> 텍스트 구성 ablation: 색깔 정보 포함 시 HitRate@5 기준 **+26%** 향상  
> Qwen3-VL이 CLIP 대비 HitRate@5 기준 **+48%** 우세

### Alpha Ablation — Hybrid Score (1,000 users 고정)

> HitRate@10에서는 alpha=0.1이 소폭 우세하나, NDCG와 MRR 전 구간에서 alpha=0.3이 최적.  
> 종합적으로 **alpha=0.3 (Emb 30% + BPR 70%)** 채택.

| alpha | HitRate@5 | HitRate@10 | NDCG@5 | NDCG@10 | MRR@5 | MRR@10 |
|-------|-----------|------------|--------|---------|-------|--------|
| 0.1 | 0.2020 | **0.2700** | 0.1362 | 0.1584 | 0.1144 | 0.1237 |
| **0.3** | **0.2160** | 0.2650 | **0.1470** | **0.1629** | **0.1243** | **0.1309** |
| 0.5 | 0.1810 | 0.2350 | 0.1215 | 0.1389 | 0.1020 | 0.1091 |
| 0.7 | 0.1230 | 0.1900 | 0.0811 | 0.1029 | 0.0674 | 0.0764 |
| 0.9 | 0.0900 | 0.1460 | 0.0585 | 0.0760 | 0.0482 | 0.0551 |

### SASRec — 시퀀스 기반 다음 구매 예측 (100 candidates, 1 positive + 99 negative)
| Model | HitRate@5 | NDCG@5 | MRR@5 | HitRate@10 | NDCG@10 | MRR@10 |
|-------|-----------|--------|-------|------------|---------|--------|
| SASRec | 0.0940 | 0.0705 | 0.0628 | 0.1090 | 0.0752 | 0.0647 |

> BPR/Embedding과 태스크가 달라 직접 비교 불가. SASRec은 시퀀스 기반 다음 구매 예측,  
> BPR/Embedding은 유저 취향 기반 추천으로 역할이 다름.  
> SASRec은 연관 추천("이런 것도 어떠세요?") 섹션에 별도 활용.
> 쿼리 맥락 반영을 위해 SASRec top50 후보에서 쿼리 임베딩 유사도로 재정렬하는 hybrid 구조 적용.
> ⚠️ 재검증 결과 `eval()` 추론 시 NaN 버그로 대부분 유저의 점수가 무효였음 → [SASRec 재검증](#sasrec-재검증-10000-users) 참고.

### Two-Tower + 재평가 (미래 구매 기준)

**요약**
- 현재 app 파이프라인 대비 **HitRate@10 +36%** (0.130 → 0.177): 가중치 튜닝 +16% → Two-Tower+Qwen 교체 +23% → 후보 확대 +36%
- BM25 코퍼스에 종류/패턴/색상 추가 시 BM25가 유효해져 **+48%** (0.130 → 0.193, 단 평가 쿼리와 단어가 일치하는 유리한 조건)
- Two-Tower+Qwen은 단독 개인화 추천에서 cornac BPR 대비 **Recall@12 +38%**, 이력 1~5개 유저 **+80%**, 신상품 추천 가능
- 성능 향상은 Two-Tower 구조가 아닌 **Qwen 콘텐츠 임베딩**에서 옴 (ID 기반 Two-Tower ≈ BPR)
- 애매한 자연어 쿼리에서는 전체 성능이 ~0.05로 떨어지고 모델 간 차이는 오차 범위 (아래 자연어 쿼리 평가)
- 후보 생성 latency 77ms → 3.8ms (BM25 전체 스코어링 제거), 쿼리 보강을 1회 호출(로컬 2B 또는 Gemini flash-lite) + 날씨 API로 바꾸면 rerank 제외 전체 **~1.5~1.6초** (현재 ~8초)
- 상품 임베딩에 이미지 추가는 텍스트 쿼리 검색·Two-Tower 모두 이득 미미 (상품 1,000개 검증)
- RRF·epsilon 탐색은 효과 없음, SASRec은 NaN 버그 발견

> 기존 평가는 train 구매 중 1개를 정답으로 써서 CF가 정답을 학습 때 이미 본 상태(**leak**).
> 새 평가(**future**)는 test 기간(시간 기준 뒤 20%)에 **새로 산 아이템**을 정답으로 사용 → CF가 못 본 미래 구매, 정답의 31%가 신상품.
> 조건: 아이템 인덱스는 Qwen3-VL-Embedding **텍스트 전용** 임베딩(105,542개). BPR은 app과 동일한 cornac BPR(k=64, 100 iter).

**모델**
- **Two-Tower**: 유저 타워(ID + 나이/멤버십) / 아이템 타워(ID + 카테고리 6종), in-batch softmax + logQ 보정
- **Two-Tower+Qwen**: 위에 아이템 타워 ← 상품 설명 Qwen 임베딩, 유저 타워 ← 최근 구매 20개 Qwen 임베딩 평균, ID dropout 30% (신상품은 ID 없이 콘텐츠로 임베딩)

#### CF 단독 비교 — test 기간 구매 예측 (10,000 users, @12)
| Model | Recall | HitRate | NDCG | 신상품 Recall |
|-------|--------|---------|------|---------------|
| Popularity (최근 1주) | 0.0092 | 0.0593 | 0.0070 | 0 |
| BPR (cornac, app) | 0.0117 | 0.0687 | 0.0104 | 0 |
| Two-Tower (ID + 카테고리) | 0.0131 | 0.0705 | 0.0112 | 0 |
| **Two-Tower+Qwen** | **0.0161** | **0.0831** | **0.0136** | **0.0021** |
| (참고) 최근 구매 재추천 | 0.0163 | 0.0777 | 0.0143 | 0 |

| 유저 이력 길이 (Recall@12) | 1-5 | 6-20 | 21+ |
|---|---|---|---|
| BPR (cornac, app) | 0.0087 | 0.0149 | 0.0106 |
| **Two-Tower+Qwen** | **0.0157** (+80%) | **0.0182** (+22%) | **0.0147** (+39%) |

> 쿼리 없이 유저 벡터 × 전체 아이템으로 test 기간 구매 예측 (홈 개인화 추천 상황).
> Two-Tower(ID+카테고리)는 BPR과 비슷한 수준 → 성능 향상은 구조가 아니라 **Qwen 콘텐츠 임베딩**에서 옴.
> Two-Tower+Qwen만 신상품 추천 가능. 서빙 비용은 BPR과 동일(64차원 내적, 후보 50개 0.003ms).

#### Hybrid — app 파이프라인 개선 (future, test 5,000 users, 가중치는 별도 tune 5,000 users에서 선택)
| 설정 | CF | 후보 | HitRate@10 | NDCG@10 | 현재 app 대비 |
|---|---|---|---|---|---|
| Embedding only | - | 쿼리 50 | 0.1044 | 0.0536 | |
| **현재 app (emb .5 / CF .3 / BM25 .2)** | BPR (cornac) | 쿼리 50 | 0.1304 | 0.0678 | 기준 |
| tuned (.4 / .6 / 0) | BPR (cornac) | 쿼리 50 | 0.1516 | 0.0803 | +16% |
| tuned (.3 / .7 / 0) | Two-Tower+Qwen | 쿼리 50 | 0.1598 | 0.0891 | +23% |
| tuned (.6 / .4 / 0) | Two-Tower+Qwen | 쿼리 50 ∪ Two-Tower 50 | 0.1692 | 0.0950 | +30% |
| **tuned (.4 / .6 / 0)** | **Two-Tower+Qwen** | **쿼리 1,000** | **0.1770** | **0.0977** | **+36%** |

> 기존 BM25 코퍼스에서는 grid search가 모든 경우 BM25 가중치 0을 선택 (향상 없음, 코퍼스 보강 시 달라짐 → [아래](#bm25-코퍼스-보강-future-test-5000-users-hitrate10)). RRF는 모든 경우 min-max보다 낮음. epsilon 탐색(0.1)은 정확도·다양성 모두 변화 미미.
> 쿼리 ∪ Two-Tower 후보에서는 쿼리 가중치가 .3 → .6으로 올라감 (쿼리와 무관한 개인화 후보를 쿼리 점수로 걸러냄).
> leak 프로토콜에서는 튜닝 가중치가 CF 0.8~0.9로 쏠림 → 정답 누수로 CF가 과대평가됨.

| 정답 종류 (HitRate@10, .3/.7/0) | BPR (cornac) | Two-Tower (ID) | Two-Tower+Qwen |
|---|---|---|---|
| 신상품 (1,510명) | 0.1040 | 0.0007 | **0.1702** |
| 기존 상품 (3,490명) | 0.1665 | **0.1799** | 0.1553 |

> Hybrid에서 Two-Tower+Qwen의 이득은 **신상품**에서 나옴. 기존 상품에서는 ID 기반 CF가 더 높음 — 쿼리 임베딩과 콘텐츠 정보가 겹치기 때문으로 추정 → 상품 종류별 CF 앙상블 여지.

| 쿼리 후보 수 (Two-Tower+Qwen) | 50 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|
| 후보 안에 정답 비율 | 28.5% | 40.4% | 53.1% | 70.3% | 82.8% |
| HitRate@10 | 0.1598 | 0.1686 | 0.1740 | 0.1766 | 0.1770 |

> 후보를 늘리면 정답 포함률은 크게 오르지만 최종 성능은 200개 이후 정체 → 가중합 재정렬의 한계, 학습 기반 랭커 여지.

#### BM25 코퍼스 보강 (future, test 5,000 users, HitRate@10)
> 기존 BM25 코퍼스(`prod_name + detail_desc`)에는 쿼리의 색상·종류 단어가 없음 (쿼리 "Vest top Solid Black" ↔ 코퍼스 "strap top jersey top with narrow shoulder straps") → BM25가 색상/종류를 매칭 못 함.
> 코퍼스에 `product_type_name + graphical_appearance_name + colour_group_name` 추가 (`BM25_CORPUS=full`).

| 설정 | 기존 코퍼스 | 보강 코퍼스 |
|---|---|---|
| BM25 only (쿼리 후보 50 재정렬) | 0.0474 | **0.1512** |
| 현재 app (.5 / .3 / .2), BPR (cornac) | 0.1304 | 0.1544 |
| tuned, BPR (cornac) | 0.1516 (.4/.6/0) | 0.1868 (.1/.3/.6) |
| **tuned, Two-Tower+Qwen** | 0.1598 (.3/.7/0) | **0.1930** (.1/.4/.5) |

> 보강 후 BM25 가중치가 0 → 0.5~0.6으로 선택되며 +20%. 단 평가 쿼리가 보강한 컬럼과 같은 단어로 만들어져 BM25에 유리한 조건 → 실제 쿼리에서는 이득이 더 작을 수 있음.

#### 자연어 쿼리 평가 (future, test 1,000 users)
> 속성 쿼리("Vest top Solid Black")는 정답을 그대로 알려줘 실제 요청보다 쉬움 → Qwen3-VL-2B-Instruct로 정답 상품마다 "유저가 했을 법한 요청"을 생성(`gen_nl_queries.py`).
> - **v1 (실패)**: 2B 모델이 "속성 1개만" 지시를 무시하고 색상·종류·상품명까지 넣음 (예: "Black high-waisted tights for a sleek, minimalist look.") → 오히려 더 쉬워짐.
> - **v2**: few-shot 예시 + 색상/패턴/상품명 단어를 코드로 제거 (예: "Comfortable socks for a casual day out", "belt for a casual day out").

| | 속성 쿼리 | 자연어 v1 | **자연어 v2** |
|---|---|---|---|
| 쿼리 후보 50 안에 정답 비율 | 28.5% | 60.8% | **10.3%** |
| Embedding only (HitRate@10) | 0.104 | 0.363 | 0.040 |
| 현재 app (.5/.3/.2), BPR (cornac) | 0.130 | 0.409 | 0.054 |
| tuned (emb + CF), Two-Tower+Qwen | 0.160 | 0.422 | 0.053 |
| 쿼리 50 ∪ Two-Tower 50, Two-Tower+Qwen | 0.169 | 0.431 | 0.058 |
| 쿼리 50 ∪ CF 50, BPR (cornac) | 0.159 | 0.417 | 0.061 |

> 애매한 쿼리(v2)에서는 정답이 쿼리 후보에 거의 안 들어와 전체 성능이 ~0.05로 떨어짐. test 1,000명이라 오차 ±0.007 → 설정·모델 간 차이는 유의하지 않음.
> 쿼리가 구체적일수록(v1) 최적 가중치가 임베딩/BM25 쪽으로 이동 → 쿼리 특성에 따라 가중치가 달라져야 함 (학습 기반 랭커 근거). 애매한 쿼리일수록 개인화 후보(∪ CF 50)가 정답 포함률을 보완.

#### 이미지 효과 검증 — 상품 1,000개 (`eval_image_1k.py`, `eval_twotower_vl_1k.py`)
> 위 실험은 모두 Qwen3-VL-Embedding에 **상품 텍스트만** 입력한 임베딩. 이미지(256×256)까지 넣은 VL 임베딩이 있는 상품 1,000개(app의 ChromaDB, 이미지 989개)로 같은 텍스트 + 이미지 유무만 다르게 비교.

| 과제 | 지표 | 텍스트만 | 텍스트 + 이미지 (VL) |
|---|---|---|---|
| 텍스트 쿼리 검색 — 속성 쿼리 | HitRate@10 | 0.819 | 0.828 |
| 텍스트 쿼리 검색 — 자연어 쿼리 | HitRate@10 | 0.413 | 0.386 |
| Two-Tower 개인화 (1,000개 카탈로그로 재학습, 21,452 users) | Recall@12 | 0.3040 | 0.3069 |

> 텍스트 쿼리 검색에서는 이미지 이득 없음 (쿼리가 텍스트라 텍스트끼리 매칭이 더 잘 맞음, 차이는 오차 범위). Two-Tower도 +1%로 오차 수준.
> 참고: 같은 1,000개에서 콘텐츠 임베딩 유무는 Recall 0.203(ID+카테고리) → 0.304로 +50% — 이득은 콘텐츠 자체에서 오고, 설명이 잘 된 데이터에서는 이미지 추가 효과가 작음.
> 한계: 1,000개는 오래된 기본 상품이라 신상품이 거의 없음 → 이미지가 가장 유리할 "설명이 부실한 신상품"·"사진 쿼리"는 별도 확인 필요.

#### SASRec 재검증 (10,000 users)
| Model | 기존 방식 재현 (1+99, 학습 target) HitRate@10 | future (전체 아이템) HitRate@10 |
|---|---|---|
| SASRec (기존 코드, 15 epoch) | 0.092 | 0.0015 |
| SASRec (버그 수정) | 0.775 | 0.0082 |
| Popularity (최근 1주) | - | 0.0083 |
| Two-Tower+Qwen | - | **0.0124** |

> **버그**: 왼쪽 패딩 시퀀스에서 패딩 위치가 attend할 key가 없어 `eval()` 추론 시 NaN → 이력 50개 미만 유저(77~89%)의 출력이 전부 NaN.
> 기존 HitRate@5 0.094는 NaN이 아닌 소수 유저(이력 50개 이상)가 학습 target을 맞힌 값 (NaN 유저는 `argsort` 결과 정답이 항상 100등). 수정: 자기 자신은 항상 attend하도록 mask + `x[:, -1]`.
> 수정 후에도 미래 구매 예측은 인기순 수준 (15 epoch, 유저당 마지막 아이템 1개만 학습) → 연관 추천에는 Two-Tower+Qwen이 우세.

#### Latency — `/recommend` 요청 1건 (`measure_latency.py`, 20건 평균, ms)
| 단계 | 현재 app | 개선안 |
|---|---|---|
| 쿼리 보강 (Gemini 대신 로컬 Qwen3-VL-2B-Instruct) | 305 | 305 |
| 쿼리 임베딩 (Qwen3-VL-Embedding, 텍스트) | 79 | 79 |
| 검색 (GPU 전수, 105k) | 2.9 (top50) | 2.7 (top1000 ∪ Two-Tower 50) |
| BM25 `get_scores` 전체 105k | 66.0 | - |
| CF 스코어링 + 가중합 + 정렬 | 8.2 (cornac BPR) | 0.8 (Two-Tower+Qwen) |
| **후보 생성 전체 (임베딩 제외)** | **77.3** | **3.8** |

> 검색은 로컬 ChromaDB(1,000개)가 아닌 GPU 전수 검색으로 측정. Reranker(3.2초)는 모델 부재로 미측정, 후보 수와 무관하게 상위 5개에만 적용.
> 쿼리 보강 305ms는 단순 번역만 한 값 (예: "출근할 때 입을 깔끔한 셔츠" → "business casual shirt"). 도시 추출·날씨 포함 전체는 아래.
> BM25를 유지한다면 `get_scores`(전체) 대신 `get_batch_scores`(후보만): 후보 50개 3.5ms.

#### Latency — 전체 요청 (`measure_e2e_latency.py`, 날씨 API 온라인, 20건 평균, ms)
> 흐름: 한국어 요청 → 로컬 Qwen3-VL-2B-Instruct 1회(영어 쿼리 + 도시 추출, JSON) → 도시가 있으면 날씨 API → 계절 키워드 추가 → 임베딩 → 검색 + Two-Tower → 상위 5. Tavily 제외.
> 날씨는 OpenWeatherMap 키가 없어 Open-Meteo(지오코딩 + 기온 2회 호출)로 대체. 8GB GPU라 LLM/임베딩을 따로 측정해 합산 (16GB면 동시 적재).

| 단계 | 평균 | p95 |
|---|---|---|
| 로컬 LLM (번역 + 도시 추출) | 1,096 | 1,492 |
| 날씨 API (도시 언급 14건) | 638 | 729 |
| 쿼리 임베딩 | 63 | 74 |
| 검색 + Two-Tower + 정렬 | 2.2 | 2.9 |
| **합계 (rerank 제외)** | **1,608** (도시 O 1,865 / 도시 X 1,009) | 2,230 |
| + Reranker (README 3.2초) | ~4,800 | |

> 현재 app(Gemini 에이전트 5~7초 + 임베딩·검색 ~1.2초, rerank 제외 ~8초) 대비 약 **5배** 빠름.
> 품질 이슈: 2B 모델이 일부 요청을 부정확하게 보강 (예: "서울 날씨에 맞는 출근룩" → "city spring", "광주 날씨 보고 니트" → 번역 실패) → 프롬프트 개선 또는 더 큰 모델 필요.

#### Latency — Gemini 쿼리 보강 (`measure_gemini_latency.py`, 요청 5건, ms)
| 모델 / 방식 | LLM 호출 | LLM | 날씨 API | 합계 |
|---|---|---|---|---|
| gemini-flash-lite, 단순 호출 (번역 + 도시 추출 1회) | 1 | **772** | 681 | **1,453** |
| gemini-flash-lite, 에이전트 (도구 선택 → 날씨 → 최종 쿼리, app 방식) | 2 | 1,611 | 642 | 2,253 |
| (참고) 로컬 Qwen3-VL-2B-Instruct, 단순 호출 | 1 | 1,096 | 638 | 1,734 |

> app의 `gemini-2.5-flash`는 신규 사용자 미제공 → flash-lite로 측정. `gemini-flash-latest`는 임시 키에서 120초 타임아웃으로 측정 불가.
> flash-lite가 로컬 2B보다 빠르고 품질도 우수 (예: "서울 날씨에 맞는 출근룩" → "office look business casual workwear spring").
> 에이전트 방식은 LLM 2회 호출로 단순 호출 대비 ~2배 → 도시 추출과 번역을 1회 호출로 합치는 것이 유리. app의 5~7초는 이전 모델 + Tavily 등 도구 호출 포함으로 추정.

#### 실험 재현
| 순서 | 스크립트 | 내용 | 시간 (8GB GPU 기준) |
|---|---|---|---|
| 1 | `make_qwen_text_emb.py` | 전체 상품 Qwen 텍스트 임베딩 → `qwen_text_emb.npy` | 30분~ |
| 2 | `make_twotower_model.py` | Two-Tower (ID + 카테고리) | 8분 |
| 3 | `make_twotower_qwen_model.py` | Two-Tower+Qwen | 15분 |
| 4 | `make_bpr_model.py` | cornac BPR (app과 동일) | 20분 |
| 5 | `compare_twotower_bpr.py` | CF 단독 비교 (+PyTorch BPR, 인기순, 재구매) | 6분 |
| 6 | `N_USERS=10000 python evaluate_hybrid_twotower.py` | Hybrid 평가 (leak/future, 가중치 tune, 구간 분석, 후보 수, latency) | 15분 |
| 7 | `evaluate_sasrec.py` | SASRec 원본 vs 수정판 | 25분 |
| (옵션) | `BM25_CORPUS=full` | 6번에서 BM25 코퍼스에 종류/패턴/색상 추가 | 10분 |
| (옵션) | `gen_nl_queries.py` → `QUERY_FILE=nl_queries.json` | Qwen3-VL-2B-Instruct로 자연어 요청 생성 후 평가 | 5분 |
| (옵션) | `gen_nl_queries.py img1k_targets.json nl_queries_1k.json` → `eval_image_1k.py nl_queries_1k.json` | 이미지 유무 검색 비교 (상품 1,000개) | 3분 |
| (옵션) | `eval_twotower_vl_1k.py` | Two-Tower 텍스트 vs VL 임베딩 (상품 1,000개) | 2분 |
| (옵션) | `measure_latency.py` | `/recommend` 단계별 latency (현재 app vs 개선안) | 5분 |
| (옵션) | `measure_e2e_latency.py` | 로컬 쿼리 보강 + 날씨 API 포함 전체 latency | 5분 |
| (옵션) | `GEMINI_API_KEY=... python measure_gemini_latency.py` | Gemini 쿼리 보강 latency (단순 호출 vs 에이전트) | 5분 |

## Roadmap
- [x] v1: 자연어 입력 → FAISS retrieval → Qwen2.5-3B-Instruct (reranking + 한국어 추천 이유)
- [x] v1: Qwen2.5-3B-Instruct KoAlpaca 기반 한국어 LoRA 파인튜닝
- [x] v1: 유저 구매 이력 임베딩 평균 기반 개인화 추천
- [x] v2: 멀티모달 전환 (Qwen3-VL-Embedding-2B + ChromaDB + 이미지)
- [x] v2: 한국어 추천 이유 생성 (Qwen3-VL-2B-Instruct)
- [x] v3: BPR 협업 필터링 + Hybrid reranking
- [x] v3: 콜드 스타트 처리 (구매 이력 없는 유저 → 임베딩 기반 fallback)
- [x] v3: CLIP vs Qwen3-VL embedding ablation 실험
- [x] v3: 날씨/일정/장소 기반 오늘의 옷 추천 (LangChain Agent)
- [x] v4: NDCG@K, MRR 평가 지표 추가
- [x] v4: SASRec 기반 시퀀스 모델 도입 → 연관 추천 섹션 활용
- [x] v4: 추천 이유 생성 → article description으로 대체 (VRAM 제약 및 latency 최적화)
- [x] v4: Reranker 128×128 최적화 (16초 → 3.2초)
- [x] v4: 인기 편향 완화 (popularity^0.01 패널티 + epsilon-greedy exploration)
- [x] v5: MLflow 실험 추적 (alpha/k 조합별 HitRate, NDCG, MRR 기록)
- [x] v5: Docker 컨테이너화 (CUDA 12.6 기반 GPU 서빙 환경 구축)
- [x] v5: Airflow DAG 기반 재학습 파이프라인 자동화 (신규 데이터 → BPR 재학습 → 평가 → 모델 교체)
- [x] v6: Two-Tower(+Qwen 콘텐츠 임베딩) 도입, cornac BPR 대비 Recall@12 +38%
- [x] v6: 미래 구매 기준 평가(future) 추가, hybrid 가중치 grid search (tune/test 분리)
- [x] v6: SASRec eval NaN 버그 발견 및 수정
- [x] v6: 후보 수 / 리트리벌(쿼리 ∪ Two-Tower) 실험, 신상품·이력 길이 구간 분석
- [x] v6: BM25 코퍼스 보강 실험 (종류/패턴/색상 추가 → BM25 유효화), 단계별 latency 측정
- [ ] v6: app에 Two-Tower+Qwen 적용, 후보 200+, BM25 코퍼스 보강 + `get_batch_scores`, 쿼리 보강 로컬화, SASRec 수정판 재학습
- [x] v6: 자연어 쿼리 평가 (LLM 생성 + 속성 단어 후처리), 날씨 API 포함 전체 latency 측정
- [ ] v6: 자연어 쿼리 평가 유저 확대 (1,000 → 5,000+, 오차 축소)
- [ ] v7: 가중합 → 학습 기반 랭커 (LightGBM), 신상품/기존상품 CF 앙상블
- [ ] v?: 실제 유저 로그인 + 히스토리 수집, Hard negative mining 적용

![데모 화면](./img/demo_image.png)
![데모_SASREC_화면](./img/next_image.png)
![mlflow_화면](./img/mlflow_image.png)
