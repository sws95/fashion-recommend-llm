import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import time
import pickle
import torch
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# /recommend 요청 1건의 단계별 latency (app_chain.py get_candidates 흐름)
# - 현재 app: cornac BPR, BM25 전체 get_scores, 후보 50, emb .5 / CF .3 / BM25 .2
# - 개선안 : Two-Tower+Qwen, BM25 제거, 쿼리 후보 1,000 (+ Two-Tower 50 합집합 옵션), emb .4 / CF .6
# 쿼리 보강은 Gemini 대신 로컬 Qwen3-VL-2B-Instruct, 검색은 ChromaDB 대신 GPU 전수 검색 (로컬 Chroma는 1,000개뿐)
# 8GB GPU라 Instruct와 Embedding 모델을 동시에 못 올림 → 쿼리 보강을 먼저 재고 해제

N_REQ  = 20
device = torch.device("cuda")
REQUESTS = ["주말 데이트에 입을 원피스 추천해줘", "출근할 때 입을 깔끔한 셔츠", "여름에 시원하게 입을 반바지",
            "운동할 때 입기 좋은 레깅스", "겨울에 따뜻한 니트", "여행 갈 때 편한 옷", "파티에 입을 화려한 드레스",
            "비 오는 날 입을 자켓", "캐주얼한 청바지 추천", "기본 흰 티셔츠"]

def sync():
    torch.cuda.synchronize()

def timed(fn):
    sync(); t = time.perf_counter(); out = fn(); sync()
    return out, (time.perf_counter() - t) * 1000

times = {}
def rec(k, v):
    times.setdefault(k, []).append(v)

# ── 1. 쿼리 보강 (로컬 Instruct) ──────────────────────────
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
proc = AutoProcessor.from_pretrained("./Qwen3-VL-2B-Instruct")
llm  = Qwen3VLForConditionalGeneration.from_pretrained("./Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()
SYSTEM = "Rewrite the user's Korean fashion request as one short English keyword query for product search. Output only the query."
enriched, examples = [], []
for i in range(N_REQ + 1):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": REQUESTS[i % len(REQUESTS)]}]}]
    ids = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out, ms = timed(lambda: llm.generate(**ids, max_new_tokens=24, do_sample=False))
    q = proc.batch_decode(out[:, ids['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
    if i > 0:                                        # 첫 요청은 warm-up
        rec('쿼리 보강 (Qwen3-VL-2B-Instruct, 로컬)', ms)
        enriched.append(q)
        examples.append((REQUESTS[i % len(REQUESTS)], q))
print("보강 예시:", examples[:5])
del llm
torch.cuda.empty_cache()

# ── 2. 모델/데이터 로딩 ───────────────────────────────────
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
emb_model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16)
INSTRUCTION = "Retrieve fashion items relevant to the query."

item_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
item_vec = torch.tensor(np.load('./qwen_text_emb.npy'), device=device)            # fp16
articles = pd.read_csv('./articles.csv', dtype={'article_id': str}).set_index('article_id').loc[item_ids]
bm25     = BM25Okapi([(str(r.prod_name) + ' ' + str(r.detail_desc)).lower().split() for r in articles.itertuples()])
bm25_pos = {a: i for i, a in enumerate(item_ids)}

with open('./bpr_model.pkl', 'rb') as f:
    b = pickle.load(f)
bpr, bds = b['model'], b['dataset']
train_user_items = b['train_interaction'].groupby('customer_id')['article_id'].apply(set).to_dict()
del b
with open('./twotower_qwen_model.pkl', 'rb') as f:
    t = pickle.load(f)
tt_user = {u: i for i, u in enumerate(t['user_ids'])}
tt_U = torch.tensor(t['user_emb'], device=device)
tt_I = torch.tensor(t['item_emb'], device=device)                                 # item_ids와 같은 순서
assert t['item_ids'] == item_ids
del t

def normalize(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

rng   = np.random.RandomState(0)
users = rng.choice([u for u in bds.uid_map if u in tt_user], N_REQ + 1, replace=False)

# ── 3. 요청 시뮬레이션 ────────────────────────────────────
for i, uid in enumerate(users):
    q = enriched[(i - 1) % len(enriched)]
    purchased = train_user_items.get(uid, set())

    q_vec, ms = timed(lambda: emb_model.process([{"text": q, "instruction": INSTRUCTION}]).to(device).half()[0])
    if i: rec('쿼리 임베딩 (Qwen3-VL-Embedding, 텍스트)', ms)

    # 현재 app ─ 검색 50 → BM25 전체 → cornac BPR → 가중합
    t0 = time.perf_counter()
    top, ms = timed(lambda: (item_vec @ q_vec).topk(50 + len(purchased)).indices)
    cand = [c for c in top.tolist() if item_ids[c] not in purchased][:50]
    if i: rec('[현재] 검색 top50 (GPU 전수, 105k)', ms)
    _, ms = timed(lambda: bm25.get_scores(q.lower().split()))
    if i: rec('[현재] BM25 get_scores 전체', ms)
    def cur_cf():
        s = bpr.score(bds.uid_map[uid])
        return np.array([s[bds.iid_map[item_ids[c]]] if item_ids[c] in bds.iid_map else s.min() for c in cand])
    bpr_s, ms = timed(cur_cf)
    if i: rec('[현재] BPR 스코어링 (cornac score)', ms)
    def cur_rank():
        e = (item_vec[cand] @ q_vec).float().cpu().numpy()
        bm = np.array(bm25.get_batch_scores(q.lower().split(), cand))
        return np.argsort(-(0.5 * normalize(e) + 0.3 * normalize(bpr_s) + 0.2 * normalize(bm)))[:5]
    _, ms = timed(cur_rank)
    if i: rec('[현재] 가중합 + 정렬', ms)
    if i: rec('[현재] 후보 생성 전체 (임베딩 제외)', (time.perf_counter() - t0) * 1000)

    # 개선안 ─ 검색 1,000 ∪ Two-Tower 50 → Two-Tower 점수 → 가중합
    t0 = time.perf_counter()
    def new_retrieve():
        qs = item_vec @ q_vec
        q_top = qs.topk(1000 + len(purchased)).indices
        u = tt_U[tt_user[uid]]
        cf_all = tt_I @ u
        t_top = cf_all.topk(50 + len(purchased)).indices
        return qs, cf_all, torch.cat([q_top, t_top]).tolist()
    (qs, cf_all, raw), ms = timed(new_retrieve)
    cand2 = [c for c in dict.fromkeys(raw) if item_ids[c] not in purchased]
    if i: rec('[개선] 검색 top1000 ∪ Two-Tower top50 (GPU)', ms)
    def new_rank():
        ct = torch.tensor(cand2, device=device)
        e = qs[ct].float().cpu().numpy(); c = cf_all[ct].float().cpu().numpy()
        return np.argsort(-(0.4 * normalize(e) + 0.6 * normalize(c)))[:5]
    _, ms = timed(new_rank)
    if i: rec('[개선] Two-Tower 스코어링 + 가중합 + 정렬', ms)
    if i: rec('[개선] 후보 생성 전체 (임베딩 제외)', (time.perf_counter() - t0) * 1000)

# ── 출력 ─────────────────────────────────────────────────
print(f"\n=== 요청 {N_REQ}건 (warm-up 1건 제외), ms ===")
print(f"{'단계':<44}{'평균':>9}{'p50':>9}{'p95':>9}")
for k, v in times.items():
    v = np.array(v)
    print(f"{k:<44}{v.mean():>9.1f}{np.median(v):>9.1f}{np.percentile(v, 95):>9.1f}")
