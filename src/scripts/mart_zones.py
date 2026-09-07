"""Витрина в разрезе зон: события по городам за неделю и месяц.

Координаты есть только у сообщений, поэтому реакции, подписки и регистрации
относятся к городу последнего сообщения того же пользователя.

Запуск:
    spark-submit --master yarn --deploy-mode cluster --py-files geo_lib.py \
        mart_zones.py <date> <depth> <events_path> <geo_path> <output_path>
"""
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from geo_lib import date_range, message_geo, read_events

# сообщения читаются за всю историю: они нужны для регистраций и для города
# пользователя, а объём у них на два порядка меньше реакций и подписок
FULL_HISTORY_DAYS = 100000


def user_home_zone(messages):
    """Город последнего сообщения пользователя."""
    latest = Window.partitionBy("user_id").orderBy(
        F.desc("event_date"), F.desc("event_ts")
    )

    return messages \
        .withColumn("rn", F.row_number().over(latest)) \
        .where("rn = 1") \
        .select("user_id", F.col("city_id").alias("zone_id"))


def registrations(messages):
    """Регистрация — первое сообщение пользователя."""
    first = Window.partitionBy("user_id").orderBy("event_date", "event_ts")

    return messages \
        .withColumn("rn", F.row_number().over(first)) \
        .where("rn = 1") \
        .select("user_id", F.col("city_id").alias("zone_id"), "event_date")


def event_zones(spark, date, depth, events_path, geo_path):
    """Все события периода с привязкой к зоне и типу."""
    start, end = date_range(date, depth)

    messages = message_geo(spark, events_path, geo_path, date, FULL_HISTORY_DAYS).cache()
    zones = F.broadcast(user_home_zone(messages))

    own = messages \
        .where(f"event_date between '{start}' and '{end}'") \
        .select(F.col("city_id").alias("zone_id"), "event_date", F.lit("message").alias("event_type"))

    new_users = registrations(messages) \
        .where(f"event_date between '{start}' and '{end}'") \
        .select("zone_id", "event_date", F.lit("user").alias("event_type"))

    events = read_events(spark, events_path, date, depth)

    reactions = events.where("event_type = 'reaction'") \
        .select(F.col("event.reaction_from").cast("long").alias("user_id"), "date") \
        .join(zones, "user_id") \
        .select("zone_id", F.col("date").alias("event_date"), F.lit("reaction").alias("event_type"))

    subscriptions = events.where("event_type = 'subscription'") \
        .select(F.col("event.user").cast("long").alias("user_id"), "date") \
        .join(zones, "user_id") \
        .select("zone_id", F.col("date").alias("event_date"), F.lit("subscription").alias("event_type"))

    return own.unionByName(new_users).unionByName(reactions).unionByName(subscriptions)


def calculate(spark, date, depth, events_path, geo_path):
    zoned = event_zones(spark, date, depth, events_path, geo_path) \
        .withColumn("month", F.trunc("event_date", "month")) \
        .withColumn("week", F.trunc("event_date", "week")) \
        .cache()

    def counters(prefix):
        return [
            F.count(F.when(F.col("event_type") == kind, 1)).alias(f"{prefix}_{kind}")
            for kind in ("message", "reaction", "subscription", "user")
        ]

    by_week = zoned.groupBy("month", "week", "zone_id").agg(*counters("week"))
    by_month = zoned.groupBy("month", "zone_id").agg(*counters("month"))

    return by_week.join(by_month, ["month", "zone_id"]).select(
        "month", "week", "zone_id",
        "week_message", "week_reaction", "week_subscription", "week_user",
        "month_message", "month_reaction", "month_subscription", "month_user",
    )


def main():
    date = sys.argv[1]
    depth = int(sys.argv[2])
    events_path = sys.argv[3]
    geo_path = sys.argv[4]
    output_path = sys.argv[5]

    spark = SparkSession.builder \
        .appName(f"MartZonesJob-{date}-d{depth}") \
        .getOrCreate()

    calculate(spark, date, depth, events_path, geo_path) \
        .write.mode("overwrite") \
        .parquet(f"{output_path}/date={date}")


if __name__ == "__main__":
    main()
