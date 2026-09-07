"""Витрина рекомендации друзей.

Пара попадает в рекомендации, если пользователи подписаны на один канал,
никогда не переписывались и находятся не дальше километра друг от друга.

Запуск:
    spark-submit --master yarn --deploy-mode cluster --py-files geo_lib.py \
        mart_friends.py <date> <depth> <events_path> <geo_path> <output_path>
"""
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from geo_lib import distance_km, message_geo, read_events

MAX_DISTANCE_KM = 1.0

# сообщения дешевле реакций и подписок на два порядка, поэтому переписка
# и координаты берутся за всю историю
FULL_HISTORY_DAYS = 100000


def user_last_position(messages):
    """Координаты, город и таймзона последнего сообщения пользователя."""
    latest = Window.partitionBy("user_id").orderBy(
        F.desc("event_date"), F.desc("event_ts")
    )

    return messages \
        .withColumn("rn", F.row_number().over(latest)) \
        .where("rn = 1") \
        .select("user_id", "lat", "lon", F.col("city_id").alias("zone_id"), "timezone")


def talked_pairs(messages_raw):
    """Пары, которые уже переписывались, в нормализованном виде."""
    pairs = messages_raw \
        .where("event.message_to is not null") \
        .select(
            F.col("event.message_from").alias("a"),
            F.col("event.message_to").alias("b"),
        )

    return pairs.select(
        F.least("a", "b").alias("user_left"),
        F.greatest("a", "b").alias("user_right"),
    ).distinct()


def channel_subscribers(spark, events_path, date, depth):
    """Уникальные пары «пользователь — канал» за период."""
    return read_events(spark, events_path, date, depth, event_type='subscription') \
        .select(
            F.col("event.user").cast("long").alias("user_id"),
            F.col("event.subscription_channel").alias("channel_id"),
        ) \
        .where("user_id is not null and channel_id is not null") \
        .distinct()


def calculate(spark, date, depth, events_path, geo_path):
    messages = message_geo(spark, events_path, geo_path, date, FULL_HISTORY_DAYS)
    positions = user_last_position(messages).cache()

    subscribers = channel_subscribers(spark, events_path, date, depth) \
        .join(F.broadcast(positions.select("user_id", "zone_id")), "user_id")

    left = subscribers.select(
        F.col("user_id").alias("user_left"), "channel_id", "zone_id"
    )
    right = subscribers.select(
        F.col("user_id").alias("user_right"), "channel_id", "zone_id"
    )

    # пара уникальна: порядок упоминания не должен порождать дубли.
    # зона в ключе join не меняет результат (ближе километра — всегда один
    # город), но сокращает число сравниваемых комбинаций
    candidates = left.join(
        right,
        ["channel_id", "zone_id"],
    ).where("user_left < user_right").select("user_left", "user_right").distinct()

    left_pos = positions.select(
        F.col("user_id").alias("user_left"),
        F.col("lat").alias("lat_left"),
        F.col("lon").alias("lon_left"),
        "zone_id", "timezone",
    )
    right_pos = positions.select(
        F.col("user_id").alias("user_right"),
        F.col("lat").alias("lat_right"),
        F.col("lon").alias("lon_right"),
    )

    close = candidates \
        .join(F.broadcast(left_pos), "user_left") \
        .join(F.broadcast(right_pos), "user_right") \
        .withColumn(
            "distance",
            distance_km(F.col("lat_left"), F.col("lon_left"),
                        F.col("lat_right"), F.col("lon_right")),
        ) \
        .where(f"distance <= {MAX_DISTANCE_KM}")

    talked = talked_pairs(
        read_events(spark, events_path, date, FULL_HISTORY_DAYS, event_type='message')
    )

    return close \
        .join(talked, ["user_left", "user_right"], "left_anti") \
        .withColumn("processed_dttm", F.current_timestamp()) \
        .withColumn(
            "local_time",
            F.from_utc_timestamp(F.col("processed_dttm"), F.col("timezone")),
        ) \
        .select("user_left", "user_right", "processed_dttm", "zone_id", "local_time")


def main():
    date = sys.argv[1]
    depth = int(sys.argv[2])
    events_path = sys.argv[3]
    geo_path = sys.argv[4]
    output_path = sys.argv[5]

    spark = SparkSession.builder \
        .appName(f"MartFriendsJob-{date}-d{depth}") \
        .getOrCreate()

    calculate(spark, date, depth, events_path, geo_path) \
        .write.mode("overwrite") \
        .parquet(f"{output_path}/date={date}")


if __name__ == "__main__":
    main()
