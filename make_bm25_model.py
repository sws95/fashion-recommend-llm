import pandas as pd
import pickle
from rank_bm25 import BM25Okapi

print("articles.csv 로딩...")
articles = pd.read_csv('./articles.csv', dtype={'article_id': str})
print(f"전체 아이템 수: {len(articles)}")

article_ids = articles['article_id'].tolist()
corpus = [
    (str(row.get('prod_name', '')) + ' ' + str(row.get('detail_desc', ''))).lower().split()
    for _, row in articles.iterrows()
]

print("BM25 인덱싱 중...")
bm25_model = BM25Okapi(corpus)

with open('./bm25_model.pkl', 'wb') as f:
    pickle.dump({
        'model':       bm25_model,
        'article_ids': article_ids,
    }, f)
print("저장 완료: ./bm25_model.pkl")