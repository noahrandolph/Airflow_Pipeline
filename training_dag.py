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
    dag_id='mock_loan_training',
    default_args=args,
    schedule_interval="@once")

# Tasks
initialize_model_db = PythonOperator(
    task_id='initialize_model_db',
    provide_context=True,
    python_callable=model_db.initialize_model_db,
    dag=dag)

load_data = PythonOperator(
    task_id='load_raw_training_data',
    provide_context=True,
    python_callable=dag_functions.load_training_data_from_db,
    dag=dag)

wrangle_data = PythonOperator(
    task_id='wrangle_training_data',
    provide_context=True,
    python_callable=dag_functions.wrangle_training_data,
    dag=dag)

train_model_1 = PythonOperator(
    task_id='train_linear_regression_model',
    provide_context=True,
    python_callable=dag_functions.train_linear_regression_model,
    dag=dag)

train_model_2 = PythonOperator(
    task_id='train_decision_tree_model',
    provide_context=True,
    python_callable=dag_functions.train_decision_tree_model,
    dag=dag)

train_model_3 = PythonOperator(
    task_id='train_random_forest_model',
    provide_context=True,
    python_callable=dag_functions.train_random_forest_model,
    dag=dag)

train_model_4 = PythonOperator(
    task_id='train_svm_model',
    provide_context=True,
    python_callable=dag_functions.train_svm_model,
    dag=dag)

select_best_model = PythonOperator(
    task_id='select_best_model',
    provide_context=True,
    python_callable=dag_functions.select_best_model,
    dag=dag)

notify = PythonOperator(
    task_id='notify',
    provide_context=True,
    python_callable=dag_functions.notify,
    dag=dag)

# Dependencies
load_data.set_upstream(initialize_model_db)
wrangle_data.set_upstream(load_data)
train_model_1.set_upstream(wrangle_data)
train_model_2.set_upstream(wrangle_data)
train_model_3.set_upstream(wrangle_data)
train_model_4.set_upstream(wrangle_data)
select_best_model.set_upstream([train_model_1, train_model_2, train_model_3, train_model_4])
notify.set_upstream(select_best_model)









