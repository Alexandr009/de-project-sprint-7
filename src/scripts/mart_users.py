"""Витрина в разрезе пользователей: актуальный и домашний город, путешествия.

Запуск:
    spark-submit --master yarn --deploy-mode cluster --py-files geo_lib.py \
        mart_users.py <date> <depth> <events_path> <geo_path> <output_path>
"""
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from geo_lib import message_geo

HOME_CITY_DAYS = 27


def user_visits(messages):
    """Непрерывные серии сообщений из одного города — «визиты».

    Смена города открывает новый визит, поэтому повторный приезд
    считается отдельным посещением.
    """
    history = Window.partitionBy("user_id").orderBy("event_date", "event_ts")
    running = history.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    return messages \
        .withColumn("prev_city", F.lag("city").over(history)) \
        .withColumn(
            "is_new_visit",
            (F.col("prev_city").isNull() | (F.col("prev_city") != F.col("city"))).cast("int"),
        ) \
        .withColumn("visit_id", F.sum("is_new_visit").over(running)) \
        .groupBy("user_id", "visit_id", "city") \
        .agg(
            F.min("event_date").alias("visit_start"),
            F.max("event_date").alias("visit_end"),
        ) \
        .withColumn("visit_days", F.datediff("visit_end", "visit_start") + 1)


def actual_city(messages):
    """Город последнего сообщения и локальное время этого сообщения."""
    latest = Window.partitionBy("user_id").orderBy(
        F.desc("event_date"), F.desc("event_ts")
    )

    return messages \
        .withColumn("rn", F.row_number().over(latest)) \
        .where("rn = 1") \
        .select(
            "user_id",
            F.col("city").alias("act_city"),
            F.from_utc_timestamp(F.col("event_ts"), F.col("timezone")).alias("local_time"),
        )


def home_city(visits):
    """Последний город, где пользователь пробыл дольше 27 дней подряд."""
    last_long_visit = Window.partitionBy("user_id").orderBy(F.desc("visit_id"))

    return visits \
        .where(f"visit_days > {HOME_CITY_DAYS}") \
        .withColumn("rn", F.row_number().over(last_long_visit)) \
        .where("rn = 1") \
        .select("user_id", F.col("city").alias("home_city"))


def travels(visits):
    """Сколько городов посещено и в каком порядке."""
    return visits \
        .groupBy("user_id") \
        .agg(
            F.count("*").alias("travel_count"),
            F.sort_array(F.collect_list(F.struct("visit_id", "city"))).alias("visit_seq"),
        ) \
        .select(
            "user_id",
            "travel_count",
            F.expr("transform(visit_seq, visit -> visit.city)").alias("travel_array"),
        )


def calculate(spark, date, depth, events_path, geo_path):
    messages = message_geo(spark, events_path, geo_path, date, depth).cache()
    visits = user_visits(messages)

    return actual_city(messages) \
        .join(home_city(visits), "user_id", "left") \
        .join(travels(visits), "user_id", "left") \
        .select(
            "user_id", "act_city", "home_city",
            "travel_count", "travel_array", "local_time",
        )


def main():
    date = sys.argv[1]
    depth = int(sys.argv[2])
    events_path = sys.argv[3]
    geo_path = sys.argv[4]
    output_path = sys.argv[5]

    spark = SparkSession.builder \
        .appName(f"MartUsersJob-{date}-d{depth}") \
        .getOrCreate()

    calculate(spark, date, depth, events_path, geo_path) \
        .write.mode("overwrite") \
        .parquet(f"{output_path}/date={date}")


if __name__ == "__main__":
    main()
