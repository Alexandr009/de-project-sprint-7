"""Общие функции для витрин геослоя: справочник городов и привязка событий к городам."""
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.sql.window import Window

EARTH_RADIUS_KM = 6371.0

# в geo.csv таймзоны нет, а для local_time она нужна: города Австралии
# распределены по шести зонам
CITY_TIMEZONE = {
    'Sydney': 'Australia/Sydney',
    'Newcastle': 'Australia/Sydney',
    'Wollongong': 'Australia/Sydney',
    'Maitland': 'Australia/Sydney',
    'Canberra': 'Australia/Canberra',
    'Melbourne': 'Australia/Melbourne',
    'Geelong': 'Australia/Melbourne',
    'Ballarat': 'Australia/Melbourne',
    'Bendigo': 'Australia/Melbourne',
    'Cranbourne': 'Australia/Melbourne',
    'Brisbane': 'Australia/Brisbane',
    'Gold Coast': 'Australia/Brisbane',
    'Townsville': 'Australia/Brisbane',
    'Cairns': 'Australia/Brisbane',
    'Toowoomba': 'Australia/Brisbane',
    'Ipswich': 'Australia/Brisbane',
    'Mackay': 'Australia/Brisbane',
    'Rockhampton': 'Australia/Brisbane',
    'Perth': 'Australia/Perth',
    'Bunbury': 'Australia/Perth',
    'Adelaide': 'Australia/Adelaide',
    'Hobart': 'Australia/Hobart',
    'Launceston': 'Australia/Hobart',
    'Darwin': 'Australia/Darwin',
}


def read_geo(spark, geo_path):
    """Справочник городов: разделитель `;`, дробная часть через запятую."""
    timezones = spark.createDataFrame(
        list(CITY_TIMEZONE.items()), ["city", "timezone"]
    )

    geo = spark.read.option("header", True).option("sep", ";").csv(geo_path) \
        .select(
            F.col("id").cast("int").alias("city_id"),
            F.col("city"),
            F.regexp_replace("lat", ",", ".").cast("double").alias("city_lat"),
            F.regexp_replace("lng", ",", ".").cast("double").alias("city_lon"),
        )

    return geo.join(F.broadcast(timezones), "city")


def distance_km(lat_1, lon_1, lat_2, lon_2):
    """Расстояние между точками на сфере по формуле гаверсинуса, км."""
    lat_1, lat_2 = F.radians(lat_1), F.radians(lat_2)
    delta_lat = (lat_2 - lat_1) / 2
    delta_lon = (F.radians(lon_2) - F.radians(lon_1)) / 2

    return F.lit(2 * EARTH_RADIUS_KM) * F.asin(F.sqrt(
        F.pow(F.sin(delta_lat), 2)
        + F.cos(lat_1) * F.cos(lat_2) * F.pow(F.sin(delta_lon), 2)
    ))


def date_range(date, depth):
    """Границы периода: `depth` дней, заканчивая датой расчёта."""
    end = datetime.strptime(date, '%Y-%m-%d')
    start = end - timedelta(days=int(depth) - 1)

    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def read_events(spark, events_path, date, depth, event_type=None):
    """События за период. Фильтр по date опускается до чтения партиций."""
    start, end = date_range(date, depth)

    events = spark.read.parquet(events_path) \
        .where(f"date between '{start}' and '{end}'")

    if event_type:
        events = events.where(f"event_type = '{event_type}'")

    return events


def with_nearest_city(events, geo):
    """Привязывает событие к ближайшему городу из справочника.

    Справочник маленький (24 города), поэтому рассылается бродкастом,
    а не шаффлится.
    """
    nearest = Window.partitionBy("event_id").orderBy("distance")

    return events \
        .where("lat is not null and lon is not null") \
        .withColumn("event_id", F.monotonically_increasing_id()) \
        .crossJoin(F.broadcast(geo)) \
        .withColumn(
            "distance",
            distance_km(F.col("lat"), F.col("lon"),
                        F.col("city_lat"), F.col("city_lon")),
        ) \
        .withColumn("rn", F.row_number().over(nearest)) \
        .where("rn = 1") \
        .drop("rn", "city_lat", "city_lon", "event_id")


def message_geo(spark, events_path, geo_path, date, depth):
    """Сообщения с определённым городом отправки.

    Год в `message_ts` личных сообщений сдвинут относительно партиции,
    поэтому хронология строится по `date`, а timestamp нужен только
    для времени суток.
    """
    geo = read_geo(spark, geo_path)
    messages = read_events(spark, events_path, date, depth, event_type='message')

    return with_nearest_city(messages, geo).select(
        F.col("event.message_from").alias("user_id"),
        F.coalesce(F.col("event.datetime"), F.col("event.message_ts"))
            .cast("timestamp").alias("event_ts"),
        F.col("date").alias("event_date"),
        "lat", "lon", "city_id", "city", "timezone",
    )
