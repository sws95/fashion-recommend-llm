import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import os
import json
import torch
import chromadb
import numpy as np
import pandas as pd
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# 쿼리에 이미지도 넣으면? — 상품 1,000개, 상품 임베딩 VL(이미지+텍스트) vs 텍스트 전용
# 쿼리 이미지 = 같은 디자인(product_code)의 다른 색상 상품 사진 ("이 옷 검정색으로" 상황, 정답 사진 그대로 쓰면 누수)
# 쿼리 4종: 텍스트만 / 이미지만 / 속성 텍스트 + 이미지 / 자연어 + 이미지
# 사용: python eval_image_query_1k.py nl_queries_1k.json

nl = json.load(open(sys.argv[1], encoding='utf-8'))
r = chromadb.PersistentClient(path='./chroma_db').get_collection('fashion_items').get(include=['embeddings', 'metadatas'])
ids = r['ids']
E_img = torch.tensor(np.array(r['embeddings'], dtype=np.float32), device='cuda')
all_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
pos = {a: i for i, a in enumerate(all_ids)}
E_txt = torch.tensor(np.load('./qwen_text_emb.npy')[[pos[a] for a in ids]].astype(np.float32), device='cuda')

art = pd.read_csv('./articles.csv', dtype={'article_id': str, 'product_code': str}).set_index('article_id').loc[ids]
img_path = lambda a: f"./images/{a[:3]}/{a}.jpg"
has_img = {a: os.path.exists(img_path(a)) for a in ids}

# 정답마다 같은 product_code의 다른 색상 상품(이미지 있는 것) 1개
targets, siblings = [], []
for code, g in art.groupby('product_code'):
    g_ids = [a for a in g.index if has_img[a]]
    for a in g.index:
        sib = [b for b in g_ids if b != a and art.loc[b, 'colour_group_name'] != art.loc[a, 'colour_group_name']]
        if sib:
            targets.append(a); siblings.append(sib[0])
N_EVAL = int(os.environ.get('N_EVAL', 400))       # 이미지 쿼리 임베딩이 느려 평가 대상 샘플링 (검색 대상은 1,000개 그대로)
if len(targets) > N_EVAL:
    keep = np.random.RandomState(42).choice(len(targets), N_EVAL, replace=False)
    targets, siblings = [targets[i] for i in keep], [siblings[i] for i in keep]
print(f"평가 대상 (다른 색상 형제 상품이 있는 상품): {len(targets)}개")

attr = {a: f"{art.loc[a, 'product_type_name']} {art.loc[a, 'graphical_appearance_name']} {art.loc[a, 'colour_group_name']}" for a in targets}
INSTRUCTION = "Retrieve fashion items relevant to the query."
queries = {
    '텍스트만 (속성)':     [{"text": attr[a]} for a in targets],
    '이미지만 (다른 색상)': [{"image": img_path(s)} for s in siblings],
    '속성 텍스트 + 이미지': [{"text": attr[a], "image": img_path(s)} for a, s in zip(targets, siblings)],
    '자연어 + 이미지':     [{"text": nl[a], "image": img_path(s)} for a, s in zip(targets, siblings)],
    # 상한선: 정답 상품 자기 사진 (정답을 그대로 보여주는 셈)
    '[상한] 정답 사진만':         [{"image": img_path(a)} if has_img[a] else {"text": attr[a]} for a in targets],
    '[상한] 속성 텍스트 + 정답 사진': [{"text": attr[a], "image": img_path(a)} if has_img[a] else {"text": attr[a]} for a in targets],
}

model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16, max_pixels=256 * 256)
idx = {a: i for i, a in enumerate(ids)}
t_idx = np.array([idx[a] for a in targets]); s_idx = np.array([idx[s] for s in siblings])
colour = art['colour_group_name'].values

rows = []
for qname, qs in queries.items():
    Q = []
    with torch.no_grad():
        for s in range(0, len(qs), 8):
            Q.append(model.process([{**q, "instruction": INSTRUCTION} for q in qs[s:s + 8]]).float())
    Q = torch.cat(Q).cuda()
    for ename, E in (('텍스트 전용', E_txt), ('이미지+텍스트(VL)', E_img)):
        top = (Q @ E.T).topk(10, dim=1).indices.cpu().numpy()
        hit = np.mean([t in row for t, row in zip(t_idx, top)])
        sib_hit = np.mean([s in row[:1] for s, row in zip(s_idx, top)])        # 쿼리 사진 상품이 1등으로 나온 비율
        col_ok = np.mean([np.mean(colour[row] == colour[t]) for t, row in zip(t_idx, top)])
        rows.append({'쿼리': qname, '상품 임베딩': ename, 'HitRate@10': hit, '요청 색상 비율@10': col_ok, '쿼리 사진 상품이 1등': sib_hit})

df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
for m in ('HitRate@10', '요청 색상 비율@10', '쿼리 사진 상품이 1등'):
    print(f"\n--- {m} ---")
    print(df.pivot(index='쿼리', columns='상품 임베딩', values=m).round(3))
