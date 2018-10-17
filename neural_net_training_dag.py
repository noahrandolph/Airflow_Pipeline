import neural_net_dag.functions.neural_net_dag_functions as nn
from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

args = {
    'owner': 'interns',
    'depends_on_past': False,
    'start_date': datetime.utcnow(),
    'email': ['noah.randolph@hcs.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    dag_id='neural_net_training',
    default_args=args,
    schedule_interval="@once")

# Tasks
load_data = PythonOperator(
    task_id='load_raw_training_data',
    provide_context=True,
    python_callable=nn.load_training_data_from_db,
    dag=dag)

wrangle_data = PythonOperator(
    task_id='wrangle_training_data',
    provide_context=True,
    python_callable=nn.wrangle_training_data,
    dag=dag)

train_model_1 = PythonOperator(
    task_id='train_wide_and_deep_neural_net_model_1',
    provide_context=True,
    python_callable=nn.train_wide_and_deep_neural_net_model_1,
    dag=dag)

train_model_2 = PythonOperator(
    task_id='train_wide_and_deep_neural_net_model_2',
    provide_context=True,
    python_callable=nn.train_wide_and_deep_neural_net_model_2,
    dag=dag)

select_best_model = PythonOperator(
    task_id='select_best_model',
    provide_context=True,
    python_callable=nn.select_best_model,
    dag=dag)

notify = PythonOperator(
    task_id='notify',
    provide_context=True,
    python_callable=nn.notify,
    dag=dag)

# Dependencies
wrangle_data.set_upstream(load_data)
train_model_1.set_upstream(wrangle_data)
train_model_2.set_upstream(wrangle_data)
select_best_model.set_upstream([train_model_1, train_model_2])
notify.set_upstream(select_best_model)









