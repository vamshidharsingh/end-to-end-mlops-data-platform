from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'customer_purchase_prediction',
    default_args=default_args,
    description='MLOps Pipeline for Propensity Prediction',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    generate_data = BashOperator(
        task_id='generate_data',
        bash_command='python /opt/airflow/scripts/data_generator.py',
    )

    run_etl = BashOperator(
        task_id='run_etl',
        bash_command='python /opt/airflow/src/transformation/etl.py',
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/src/training/train.py',
    )

    generate_data >> run_etl >> train_model
