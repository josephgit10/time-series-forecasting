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

    forecast = BashOperator(
        task_id='run_forecast',
        bash_command='python /app/src/forecasting_pipeline.py'
    )

    preprocess >> forecast
