import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import os
import torch
import chromadb
import numpy as np
import pandas as pd
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# 쿼리 종류별 "정답에 얼마나 가까워지나" — 상품 N개, 1,000개 중 정답 순위
# 쿼리 사진 = 같은 디자인(product_code)의 다른 색상 상품, 상한 = 정답 상품 자기 사진
# 사용: python demo_image_query.py 10        (N개 랜덤, 디자인당 1개)
#       python demo_image_query.py 0108775015 (특정 상품)

r = chromadb.PersistentClient(path='./chroma_db').get_collection('fashion_items').get(include=['embeddings'])
ids = r['ids']
E = {'VL': torch.tensor(np.array(r['embeddings'], dtype=np.float32), device='cuda')}
all_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
pos = {a: i for i, a in enumerate(all_ids)}
E['텍스트'] = torch.tensor(np.load('./qwen_text_emb.npy')[[pos[a] for a in ids]].astype(np.float32), device='cuda')

art = pd.read_csv('./articles.csv', dtype={'article_id': str, 'product_code': str}).set_index('article_id').loc[ids]
img = lambda a: f"./images/{a[:3]}/{a}.jpg"
name = lambda a: f"{art.loc[a, 'prod_name']} / {art.loc[a, 'colour_group_name']}"

def sibling(a):
    g = art[art['product_code'] == art.loc[a, 'product_code']]
    s = [b for b in g.index if b != a and os.path.exists(img(b)) and g.loc[b, 'colour_group_name'] != art.loc[a, 'colour_group_name']]
    return s[0] if s else None

arg = sys.argv[1] if len(sys.argv) > 1 else '1'
if len(arg) < 5:
    cands = [a for a in ids if os.path.exists(img(a)) and sibling(a)]
    cands = list({art.loc[a, 'product_code']: a for a in cands}.values())
    targets = list(np.random.RandomState(0).choice(cands, int(arg), replace=False))
else:
    targets = [arg]

model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16, max_pixels=256 * 256)
INSTRUCTION = "Retrieve fashion items relevant to the query."
ORDER = ['1 텍스트만', '2 다른 색상 사진만', '3 텍스트 + 다른 색상 사진', '4 자연어 + 다른 색상 사진',
         '5 [상한] 정답 사진만', '6 [상한] 텍스트 + 정답 사진']
import json
NL = json.load(open('./nl_queries_1k.json', encoding='utf-8'))    # gen_nl_queries.py (색상·패턴·상품명 제거된 자연어 요청)

rows = []
for target in targets:
    sib = sibling(target)
    text = f"{art.loc[target, 'product_type_name']} {art.loc[target, 'graphical_appearance_name']} {art.loc[target, 'colour_group_name']}"
    print(f"정답: {name(target)} | 쿼리 사진: {name(sib)} | 텍스트: \"{text}\"", flush=True)
    queries = dict(zip(ORDER, [{"text": text}, {"image": img(sib)}, {"text": text, "image": img(sib)},
                               {"text": NL[target], "image": img(sib)},
                               {"image": img(target)}, {"text": text, "image": img(target)}]))
    t = ids.index(target)
    for qname, q in queries.items():
        with torch.no_grad():
            qv = model.process([{**q, "instruction": INSTRUCTION}]).float().cuda()[0]
        for ename, M in E.items():
            s = M @ qv
            rows.append({'정답': name(target), '쿼리': qname, '상품': ename, '순위': int((s > s[t]).sum()) + 1})

df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
print("\n=== 정답 순위 (1,000개 중, 상품 임베딩 VL / 텍스트) ===")
print(df.pivot_table(index='정답', columns=['쿼리', '상품'], values='순위')[ORDER].to_string())
print("\n=== 요약 ===")
summ = df.groupby(['쿼리', '상품'])['순위'].agg(중앙값='median', top10비율=lambda v: (v <= 10).mean())
print(summ.reindex(ORDER, level=0).round(2).to_string())
