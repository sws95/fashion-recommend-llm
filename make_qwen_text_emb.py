import sys
sys.path.append("./Qwen3-VL-Embedding-2B")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# 투타워 학습용: 전체 아이템 텍스트 전용 Qwen 임베딩 (이미지 제외 → 빠름)
model = Qwen3VLEmbedder(
    model_name_or_path="./Qwen3-VL-Embedding-2B",
    torch_dtype=torch.bfloat16,
)
print("모델 로딩 완료")

INSTRUCTION = "Retrieve fashion items relevant to the query."
BATCH_SIZE  = 64

articles = pd.read_csv('./articles.csv', dtype={'article_id': str})
texts = (
    articles['prod_name'] + " " + articles['product_type_name'] + " " +
    articles['colour_group_name'] + " " + articles['detail_desc'].fillna('')
).tolist()

embs = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="임베딩"):
        inputs = [{"text": t, "instruction": INSTRUCTION} for t in texts[i:i + BATCH_SIZE]]
        embs.append(model.process(inputs).float().cpu().numpy().astype(np.float16))

np.save('./qwen_text_emb.npy', np.concatenate(embs))
articles[['article_id']].to_csv('./qwen_text_emb_ids.csv', index=False)
print(f"저장 완료: ./qwen_text_emb.npy {len(texts)}개")
