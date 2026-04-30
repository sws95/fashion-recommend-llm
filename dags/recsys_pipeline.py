# dags/recsys_pipeline.py

#default_args = {
#    'owner': 'airflow',
#    'retries': 1,
#    'retry_delay': timedelta(minutes=5),
#}
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import pandas as pd
import os

default_args = {
    'owner': 'airflow',
    'retries': 0,
}

def check_new_data():
    """새 거래 데이터 확인"""
    transactions = pd.read_csv('/opt/airflow/transactions_train.csv', dtype={
        'customer_id': str, 'article_id': str
    })
    print(f"총 거래 수: {len(transactions)}")
    return len(transactions)

def retrain_bpr():
    """BPR 모델 재학습"""
    import cornac
    import pickle
    import pandas as pd
    import numpy as np

    transactions = pd.read_csv('/opt/airflow/transactions_train.csv', dtype={
        'customer_id': str, 'article_id': str
    })
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    split_date = transactions['t_dat'].quantile(0.8)
    train_df = transactions[transactions['t_dat'] < split_date]
    train_interaction = train_df.groupby(['customer_id', 'article_id']).size().reset_index(name='count')

    # BPR 학습
    dataset = cornac.data.Dataset.from_uir(
        train_interaction[['customer_id', 'article_id', 'count']].itertuples(index=False),
        seed=42
    )
    bpr = cornac.models.BPR(k=64, max_iter=200, learning_rate=0.01, seed=42)
    bpr.fit(dataset)

    idx2item = {v: k for k, v in dataset.iid_map.items()}

    with open('/opt/airflow/bpr_model_new.pkl', 'wb') as f:
        pickle.dump({
            'model': bpr,
            'dataset': dataset,
            'idx2item': idx2item,
            'train_interaction': train_interaction
        }, f)
    print("BPR 재학습 완료")

def evaluate_model():
    """신규 모델 평가"""
    import pickle
    import numpy as np
    import pandas as pd

    with open('/opt/airflow/bpr_model_new.pkl', 'rb') as f:
        bpr_data = pickle.load(f)

    bpr_model   = bpr_data['model']
    bpr_dataset = bpr_data['dataset']
    idx2item    = bpr_data['idx2item']
    train_interaction = bpr_data['train_interaction']
    train_user_items = train_interaction.groupby('customer_id')['article_id'].apply(set).to_dict()

    valid_users = sorted([
        uid for uid, items in train_user_items.items()
        if len(items) >= 2 and uid in bpr_dataset.uid_map
    ])
    np.random.seed(42)
    fixed_users = np.random.choice(valid_users, min(500, len(valid_users)), replace=False)
    np.random.seed(42)
    user_targets = {
        uid: np.random.choice(list(sorted(train_user_items[uid])))
        for uid in fixed_users
    }

    hits = []
    for uid in fixed_users:
        target = user_targets[uid]
        remaining = train_user_items[uid] - {target}
        user_idx = bpr_dataset.uid_map[uid]
        item_scores = bpr_model.score(user_idx)

        recommended = []
        for idx in np.argsort(item_scores)[::-1]:
            aid = idx2item[idx]
            if aid not in remaining:
                recommended.append(aid)
            if len(recommended) == 10:
                break
        hits.append(1.0 if target in recommended else 0.0)

    hitrate = np.mean(hits)
    print(f"신규 모델 HitRate@10: {hitrate:.4f}")

    # 기준 이상이면 모델 교체
    if hitrate >= 0.10:
        import shutil
        shutil.copy('/opt/airflow/bpr_model_new.pkl', '/opt/airflow/bpr_model.pkl')
        print("모델 교체 완료")
    else:
        print(f"성능 미달 ({hitrate:.4f} < 0.10), 모델 유지")

with DAG(
    dag_id="recsys_pipeline",
    default_args=default_args,
    description="RecSys 재학습 파이프라인",
    schedule_interval="0 2 * * *",  # 매일 새벽 2시
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="check_new_data",
        python_callable=check_new_data,
    )

    t2 = PythonOperator(
        task_id="retrain_bpr",
        python_callable=retrain_bpr,
    )

    t3 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t1 >> t2 >> t3