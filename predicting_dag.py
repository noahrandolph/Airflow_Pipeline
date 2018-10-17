import intern_machine_learning_framework.functions.dag_functions as dag_functions
import intern_machine_learning_framework.functions.model_db_interface as model_db
from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

args = {
    'owner': 'noah',
    'depends_on_past': False,
    'start_date': datetime.utcnow(),
    'email': ['noah.randolph@hcs.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    dag_id='mock_loan_predicting',
    default_args=args,
    schedule_interval="@once")

# Tasks
load_unseen_data = PythonOperator(
    task_id='load_raw_unseen_prediction_data',
    provide_context=True,
    python_callable=dag_functions.load_unseen_prediction_data_from_db,
    dag=dag)

wrangle_unseen_data = PythonOperator(
    task_id='wrangle_unseen_data',
    provide_context=True,
    python_callable=dag_functions.wrangle_unseen_data,
    dag=dag)

predict_from_unseen_data = PythonOperator(
    task_id='predict_from_unseen_data',
    provide_context=True,
    python_callable=dag_functions.predict_from_unseen_data,
    dag=dag)

store_predictions_as_i_tags = PythonOperator(
    task_id='store_predictions_as_i_tags',
    provide_context=True,
    python_callable=dag_functions.store_predictions_as_i_tags,
    dag=dag)

# Dependencies
wrangle_unseen_data.set_upstream(load_unseen_data)
predict_from_unseen_data.set_upstream(wrangle_unseen_data)
store_predictions_as_i_tags.set_upstream([predict_from_unseen_data])





