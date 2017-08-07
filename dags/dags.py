from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators import PythonOperator
from build_models import build_models
from download_data import download_data
from transform_data import transform_data
from validate_models import validate_models


def make_task(func):
    return PythonOperator(
        task_id=func.__name__,
        provide_context=True,
        python_callable=func,
        dag=dag,
    )

    
args = {
    'owner': 'airflow',
    'start_date': datetime(2017, 8, 6, 10, 0, 0),
}

dag = DAG(
    dag_id='fantasy_football',
    default_args=args,
    schedule_interval=timedelta(days=1),
)


# define tasks
import_task = make_task(download_data)
transform_task = make_task(transform_data)
model_task = make_task(build_models)
validate_task = make_task(validate_models)

# define dependencies
transform_task.set_upstream(import_task)
model_task.set_upstream(transform_task)
validate_task.set_upstream(transform_task)
