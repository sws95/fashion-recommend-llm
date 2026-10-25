import sys
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 정답 상품마다 "유저가 했을 법한 자연어 요청"을 생성 → evaluate_hybrid_twotower.py QUERY_FILE로 평가
# 속성 쿼리("Vest top Solid Black")는 정답을 그대로 알려줘서 너무 쉬움 → 상황/스타일 위주로 덜 직접적으로
# 사용: python gen_nl_queries.py targets.json nl_queries.json

targets_path, out_path = sys.argv[1], sys.argv[2]
targets = json.load(open(targets_path))

articles = pd.read_csv('./articles.csv', dtype={'article_id': str}).set_index('article_id')

MODEL = "./Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL)
processor.tokenizer.padding_side = 'left'
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

PROMPT = """You are a shopper looking for clothes. Below is a product you would end up buying.
Write ONE short request (max 15 words) you would type into a fashion shopping assistant to find something like it.
Rules: describe the occasion, season, vibe or how you'd wear it. Mention at most ONE concrete attribute (color OR item type). Do not copy the product name. Output only the request.

Product: {name} | type: {ptype} | color: {colour} | pattern: {pattern} | section: {section}
Description: {desc}"""

def build(aid):
    r = articles.loc[aid]
    text = PROMPT.format(name=r['prod_name'], ptype=r['product_type_name'], colour=r['colour_group_name'],
                         pattern=r['graphical_appearance_name'], section=r['index_group_name'],
                         desc=r['detail_desc'] if pd.notna(r['detail_desc']) else '')
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

BATCH = 32
out = {}
with torch.no_grad():
    for s in tqdm(range(0, len(targets), BATCH), desc="쿼리 생성", mininterval=10):
        batch = targets[s:s + BATCH]
        inputs = processor.tokenizer([build(a) for a in batch], return_tensors='pt', padding=True).to('cuda')
        gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        texts = processor.tokenizer.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        for a, t in zip(batch, texts):
            out[a] = t.strip().strip('"').split('\n')[0]

json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(f"저장: {out_path} ({len(out)}개)")
for a in targets[:8]:
    print(f"  {articles.loc[a, 'prod_name']} ({articles.loc[a, 'product_type_name']}, {articles.loc[a, 'colour_group_name']}) → {out[a]}")
