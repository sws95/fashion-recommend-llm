import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import os
import torch
import chromadb
import pandas as pd
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# 모델 로딩
model = Qwen3VLEmbedder(
    model_name_or_path="./Qwen3-VL-Embedding-2B",
    torch_dtype=torch.bfloat16,
    max_pixels=256*256,
)
print("모델 로딩 완료")

INSTRUCTION = "Retrieve fashion items relevant to the query."

# ChromaDB 세팅
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("fashion_items_v2")
    print("기존 collection 삭제")
except:
    pass

collection = client.create_collection(
    name="fashion_items_v2",
    metadata={"hnsw:space": "cosine"}
)

def get_image_path(article_id):
    folder = str(article_id)[:3]
    return f"./images/{folder}/{article_id}.jpg"

def print_vram():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM | 사용중: {allocated:.2f}GB | 예약됨: {reserved:.2f}GB")

articles = pd.read_csv('./articles.csv', dtype={'article_id': str})
print(f"전체 아이템 수: {len(articles)}")
batch_size = 4

total_with_image = 0
total_without_image = 0

for i in range(0, len(articles), batch_size):
    batch = articles.iloc[i:i+batch_size]

    inputs = []
    for _, row in batch.iterrows():
        text = (
            f"{row['prod_name']} {row['product_type_name']} "
            f"{row['colour_group_name']} "
            f"{row['detail_desc'] if pd.notna(row['detail_desc']) else ''}"
        )
        img_path = get_image_path(row['article_id'])
        inp = {"text": text, "instruction": INSTRUCTION}
        if os.path.exists(img_path):
            inp["image"] = img_path
            total_with_image += 1
        else:
            total_without_image += 1
        inputs.append(inp)

    embeddings = model.process(inputs)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=[inp["text"] for inp in inputs],
        metadatas=[{
            "article_id": str(row['article_id']),
            "prod_name": row['prod_name'],
            "has_image": os.path.exists(get_image_path(row['article_id']))
        } for _, row in batch.iterrows()],
        ids=[str(row['article_id']) for _, row in batch.iterrows()]
    )

    if i % 10 == 0:
        print(f"{i}/{len(articles)} 완료 | 이미지 있음: {total_with_image} | 없음: {total_without_image}")
        print_vram()

print(f"\n저장된 아이템 수: {collection.count()}")
print(f"이미지 있음: {total_with_image} | 없음: {total_without_image}")
print_vram()