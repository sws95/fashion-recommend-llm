import pandas as pd
import numpy as np
import cornac
from cornac.data import Dataset
from cornac.models import BPR
import pickle

print("데이터 로딩...")
transactions = pd.read_csv('./transactions_train.csv', dtype={
    'customer_id': str,
    'article_id': str
})
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

split_date = transactions['t_dat'].quantile(0.8)
train_df = transactions[transactions['t_dat'] < split_date]

train_interaction = train_df.groupby(['customer_id', 'article_id']).size().reset_index(name='count')
train_data = list(zip(
    train_interaction['customer_id'],
    train_interaction['article_id'],
    train_interaction['count'].astype(float)
))

train_dataset = Dataset.from_uir(train_data)
idx2item = {v: k for k, v in train_dataset.iid_map.items()}

print("BPR 학습 시작...")
model = BPR(
    k=64,
    max_iter=100,
    learning_rate=0.01,
    lambda_reg=0.01,
    verbose=True,
    seed=42
)
model.fit(train_dataset)
print("BPR 학습 완료")

with open('./bpr_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'dataset': train_dataset,
        'idx2item': idx2item,
        'train_interaction': train_interaction,
    }, f)
print("모델 저장 완료: ./bpr_model.pkl")