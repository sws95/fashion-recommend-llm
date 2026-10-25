import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import json
import torch
import chromadb
import numpy as np
import pandas as pd
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# 이미지가 유리한가? — 같은 상품 1,000개, 같은 텍스트로 만든 임베딩 2종 비교
#   이미지+텍스트: ChromaDB fashion_items (make_embedding_qwen.py, 256×256, app에서 쓰던 것)
#   텍스트 전용  : qwen_text_emb.npy 앞 1,000개 (make_qwen_text_emb.py)
# 과제: 상품마다 쿼리로 1,000개 중 그 상품 찾기
#   HitRate@10 = 정답 상품이 top10 안 / 관련도@10 = top10 중 종류·색상이 정답과 같은 비율
# 쿼리: 속성("Vest top Solid Black") / 자연어 v2(gen_nl_queries.py, 색상·패턴·상품명 제거)
# 사용: python eval_image_1k.py nl_queries_1k.json

nl_path = sys.argv[1]
col = chromadb.PersistentClient(path='./chroma_db').get_collection('fashion_items')
r = col.get(include=['embeddings', 'metadatas'])
ids = r['ids']
has_img = np.array([m['has_image'] for m in r['metadatas']])
E_img = torch.tensor(np.array(r['embeddings'], dtype=np.float32), device='cuda')

all_ids = pd.read_csv('./qwen_text_emb_ids.csv', dtype={'article_id': str})['article_id'].tolist()
pos = {a: i for i, a in enumerate(all_ids)}
E_txt = torch.tensor(np.load('./qwen_text_emb.npy')[[pos[a] for a in ids]].astype(np.float32), device='cuda')

art = pd.read_csv('./articles.csv', dtype={'article_id': str}).set_index('article_id').loc[ids]
ptype, colour = art['product_type_name'].values, art['colour_group_name'].values
nl = json.load(open(nl_path, encoding='utf-8'))
queries = {
    '속성 쿼리': [f"{r.product_type_name} {r.graphical_appearance_name} {r.colour_group_name}" for r in art.itertuples()],
    '자연어 쿼리': [nl[a] for a in ids],
}

model = Qwen3VLEmbedder(model_name_or_path="./Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16)
INSTRUCTION = "Retrieve fashion items relevant to the query."

rows = []
for qname, qs in queries.items():
    Q = []
    with torch.no_grad():
        for s in range(0, len(qs), 64):
            Q.append(model.process([{"text": t, "instruction": INSTRUCTION} for t in qs[s:s + 64]]).float())
    Q = torch.cat(Q).cuda()
    for ename, E in (('텍스트 전용', E_txt), ('이미지+텍스트', E_img)):
        top = (Q @ E.T).topk(10, dim=1).indices.cpu().numpy()
        hit = np.array([i in top[i] for i in range(len(ids))])
        rel = np.array([np.mean((ptype[top[i]] == ptype[i]) & (colour[top[i]] == colour[i])) for i in range(len(ids))])
        for subset, m in (('전체', np.ones(len(ids), bool)), ('이미지 있는 상품', has_img)):
            rows.append({'쿼리': qname, '임베딩': ename, '대상': subset, 'HitRate@10': hit[m].mean(), '관련도@10': rel[m].mean()})

df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(f"\n=== 상품 1,000개 (이미지 {has_img.sum()}개) ===")
print(df[df['대상'] == '전체'].pivot(index='쿼리', columns='임베딩', values=['HitRate@10', '관련도@10']).round(3))
print("\n예시 자연어 쿼리:", [nl[a] for a in ids[:5]])
