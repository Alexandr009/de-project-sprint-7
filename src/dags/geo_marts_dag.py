"""Ежедневное обновление витрин геослоя.

Витрины считаются независимо друг от друга: каждая читает сырые события
из Raw-слоя и пишет свою партицию в слой аналитики.
"""
import os
from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['JAVA_HOME'] = '/usr'
os.environ['SPARK_HOME'] = '/usr/lib/spark'
os.environ['PYTHONPATH'] = '/usr/local/lib/python3.8'

USER = 's19239739'
SCRIPTS_PATH = '/lessons'
EVENTS_PATH = '/user/master/data/geo/events'
GEO_PATH = f'/user/{USER}/data/geo/geo.csv'
ANALYTICS_PATH = f'/user/{USER}/data/analytics'

# глубина расчёта в днях: витрина пользователей строится на всей истории,
# витрина зон — за месяц, рекомендации — за неделю подписок
USERS_DEPTH = '180'
ZONES_DEPTH = '30'
FRIENDS_DEPTH = '7'

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022, 6, 21),
    'retries': 1,
}

dag_spark = DAG(
    dag_id='geo_marts',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    description='Витрины геослоя: пользователи, зоны, рекомендации друзей',
)

spark_conf = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
}


def spark_job(task_id, script, depth, output_dir, shuffle_partitions=None):
    conf = dict(spark_conf)
    if shuffle_partitions:
        conf["spark.sql.shuffle.partitions"] = shuffle_partitions

    return SparkSubmitOperator(
        task_id=task_id,
        dag=dag_spark,
        application=f'{SCRIPTS_PATH}/{script}',
        conn_id='yarn_spark',
        py_files=f'{SCRIPTS_PATH}/geo_lib.py',
        application_args=[
            '{{ ds }}',
            depth,
            EVENTS_PATH,
            GEO_PATH,
            f'{ANALYTICS_PATH}/{output_dir}',
        ],
        conf=conf,
        executor_cores=2,
        executor_memory='4g',
        num_executors=3,
    )


mart_users = spark_job('mart_users', 'mart_users.py', USERS_DEPTH, 'mart_users')
mart_zones = spark_job('mart_zones', 'mart_zones.py', ZONES_DEPTH, 'mart_zones')
mart_friends = spark_job(
    'mart_friends', 'mart_friends.py', FRIENDS_DEPTH, 'mart_friends',
    shuffle_partitions='400',
)

[mart_users, mart_zones, mart_friends]
