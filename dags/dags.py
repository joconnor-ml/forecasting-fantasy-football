from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators import PythonOperator, LatestOnlyOperator
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
    'start_date': datetime(2018, 8, 9, 10, 0, 0),
}

dag = DAG(
    dag_id='fantasy_football',
    default_args=args,
    schedule_interval=timedelta(days=1),
)


# define tasks
latest_task = LatestOnlyOperator(
    task_id="latest_only",
    dag=dag,
)
import_task = make_task(download_data)
transform_task = make_task(transform_data)
model_task = make_task(build_models)
validate_task = make_task(validate_models)

latest_task >> import_task >> transform_task >> model_task >> validate_task
