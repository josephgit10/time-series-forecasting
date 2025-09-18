from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='daily_forecast_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    preprocess = BashOperator(
        task_id='preprocess_data',
        bash_command='python /app/src/preprocessing.py'
    )

    models = ['prophet', 'xgboost', 'lstm']
    forecast_tasks = []

    for model in models:
        forecast_task = BashOperator(
            task_id=f'run_forecast_{model}',
            bash_command=f'python /app/src/forecasting_pipeline.py --model {model}'
        )
        forecast_tasks.append(forecast_task)

    preprocess >> forecast_tasks
