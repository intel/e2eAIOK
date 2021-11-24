from pyspark.context import SparkContext, SparkConf
from pyspark.sql.functions import col, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, TimestampType, FloatType, ArrayType, DoubleType
import datetime
import hashlib
import math
import time
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.linalg import SparseVector, VectorUDT
import argparse
import tensorflow as tf
from pyspark import TaskContext
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from data.outbrain.spark.utils.feature_description import PREBATCH_SIZE, HASH_BUCKET_SIZES
from data.outbrain.spark.utils.feature_description import LABEL_COLUMN, DISPLAY_ID_COLUMN, CATEGORICAL_COLUMNS, \
    DOC_CATEGORICAL_MULTIVALUED_COLUMNS, BOOL_COLUMNS, INT_COLUMNS, FLOAT_COLUMNS, \
    FLOAT_COLUMNS_LOG_BIN_TRANSFORM, FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM, FLOAT_COLUMNS_NO_TRANSFORM

import os
import threading

os.environ["PYSPARK_PYTHON"] = "/root/sw/miniconda3/envs/spark/bin/python"

OUTPUT_BUCKET_FOLDER = "/tmp/spark/preprocessed/rewrite/"
DATA_BUCKET_FOLDER = "/outbrain/orig/"
SPARK_TEMP_FOLDER = "/tmp/spark/spark-temp/"
TENSORFLOW_HADOOP = "data/outbrain/spark/data/tensorflow-hadoop-1.5.0.jar"


conf = SparkConf().setMaster('spark://sr112:7077').set('spark.executor.memory', '40g').set('spark.driver.memory', '50g').set('spark.executor.cores', '8')
conf.set("spark.jars", TENSORFLOW_HADOOP)
conf.set("spark.sql.files.maxPartitionBytes", 805306368)
conf.set("spark.memory.offHeap.enabled", True)
conf.set("spark.memory.offHeap.size", "5g")
conf.set("spark.sql.shuffle.partitions", 2000)
conf.set("spark.sql.adaptive.enabled", True)
conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
conf.set("spark.sql.adaptive.skewJoin.enabled", True)


sc = SparkContext(conf=conf)
spark = SparkSession(sc)

clock = time.time()

LESS_SPECIAL_CAT_VALUE = 'less'

def convert_odd_timestamp(timestamp_ms_relative):
    TIMESTAMP_DELTA = 1465876799998
    return datetime.datetime.fromtimestamp((int(timestamp_ms_relative) + TIMESTAMP_DELTA) // 1000)

INT_DEFAULT_NULL_VALUE = -1
int_null_to_minus_one_udf = F.udf(lambda x: x if x is not None else INT_DEFAULT_NULL_VALUE, IntegerType())
int_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(IntegerType()))
float_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(FloatType()))
str_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(StringType()))

events_schema = StructType(
    [StructField("display_id", IntegerType(), True),
     StructField("uuid_event", StringType(), True),
     StructField("document_id_event", IntegerType(), True),
     StructField("timestamp_event", IntegerType(), True),
     StructField("platform_event", IntegerType(), True),
     StructField("geo_location_event", StringType(), True)]
)

events_df = spark.read.schema(events_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "events.csv") \
    .withColumn('day_event', (F.col('timestamp_event')/ 1000 / 60 / 60 / 24).cast(IntegerType())) \
    .withColumn('event_country', F.substring('geo_location_event', 0, 2)) \
    .withColumn('event_country_state', F.substring('geo_location_event', 0, 5)) \
    .alias('events')

print('Drop rows with empty "geo_location"...')
events_df = events_df.dropna(subset="geo_location_event")

print('Drop rows with empty "platform"...')
events_df = events_df.dropna(subset="platform_event")

promoted_content_schema = StructType(
    [StructField("ad_id", IntegerType(), True),
     StructField("document_id_promo", IntegerType(), True),
     StructField("campaign_id", IntegerType(), True),
     StructField("advertiser_id", IntegerType(), True)]
)

promoted_content_df = spark.read.schema(promoted_content_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "promoted_content.csv") \
    .alias('promoted_content')

clicks_train_schema = StructType(
    [StructField("display_id", IntegerType(), True),
     StructField("ad_id", IntegerType(), True),
     StructField("clicked", IntegerType(), True)]
)

clicks_train_df = spark.read.schema(clicks_train_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "clicks_train.csv") \
    .alias('clicks_train')

documents_meta_schema = StructType(
    [StructField("document_id_doc", IntegerType(), True),
     StructField("source_id", IntegerType(), True),
     StructField("publisher_id", IntegerType(), True),
     StructField("publish_time", TimestampType(), True)]
)

documents_meta_df = spark.read.schema(documents_meta_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_meta.csv") \
    .alias('documents_meta')

# Drop rows with empty "source_id"
documents_meta_df = documents_meta_df.dropna(subset="source_id")

source_publishers_df = documents_meta_df.select(["source_id", "publisher_id"]).dropDuplicates()

# get list of source_ids without publisher_id
rows_no_pub = source_publishers_df.filter("publisher_id is NULL")
source_ids_without_publisher = sorted([row['source_id'] for row in rows_no_pub.collect()])

# maximum value of publisher_id used so far
max_pub = max(source_publishers_df.select(["publisher_id"]).dropna().collect())['publisher_id']

# rows filled with new publisher_ids
new_publishers = [(source, max_pub + 1 + nr) for nr, source in enumerate(source_ids_without_publisher)]
new_publishers_df = spark.createDataFrame(new_publishers, ("source_id", "publisher_id"))

# old and new publishers merged
fixed_source_publishers_df = source_publishers_df.dropna().union(new_publishers_df)

# update documents_meta with bew publishers
documents_meta_df = documents_meta_df.drop('publisher_id').join(fixed_source_publishers_df, on='source_id')

documents_total = documents_meta_df.count()


events_joined_df = events_df.join(documents_meta_df
                                  .withColumnRenamed('source_id', 'source_id_doc_event')
                                  .withColumnRenamed('publisher_id', 'publisher_doc_event')
                                  .withColumnRenamed('publish_time', 'publish_time_doc_event')
                                  .withColumnRenamed('document_id_doc', 'document_id_doc_event'),
                                  on=F.col("document_id_event") == F.col("document_id_doc_event"), how='left').alias('events')


clicks_train_joined_df = clicks_train_df \
    .join(promoted_content_df, on='ad_id', how='left') \
    .join(documents_meta_df, on=F.col("promoted_content.document_id_promo") == F.col("documents_meta.document_id_doc"), how='left') \
    .join(events_joined_df, on='display_id', how='left')

clicks_train_joined_df.createOrReplaceTempView('clicks_train_joined')


validation_display_ids_df = clicks_train_joined_df.select('display_id', 'day_event').distinct() \
    .sampleBy("day_event", fractions={0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2,
                                      5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.2, 10: 0.2, 11: 1.0, 12: 1.0}, seed=0)

valid_id = validation_display_ids_df.select('display_id').distinct().createOrReplaceTempView("validation_display_ids")

valid_set_df = spark.sql('''
SELECT * FROM clicks_train_joined t
WHERE EXISTS (SELECT display_id FROM validation_display_ids
WHERE display_id = t.display_id)''')

s_time = time.time()
valid_set_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'valid_set_df')
valid_set_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'valid_set_df')
print(f'valid_set_df time: {time.time() - s_time}')

train_set_df = spark.sql('''
SELECT * FROM clicks_train_joined t
WHERE NOT EXISTS (SELECT display_id FROM validation_display_ids
WHERE display_id = t.display_id)''')

s_time = time.time()
train_set_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_set_df')
train_set_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_set_df')
print(f'train_set_df time: {time.time() - s_time}')

print(f'train/test dataset generation time: {time.time() - clock}')
clock = time.time()

################################################################################################################
################################################################################################################
# event_country_values_counts = get_category_field_values_counts('event_country', events_df, min_threshold=10)
country_value_cat = events_df.select('event_country').groupBy('event_country').count().filter('event_country is not null and count >= 10')

# event_country_state_values_counts = get_category_field_values_counts('event_country_state', events_df, min_threshold=10)
state_value_cal = events_df.select('event_country_state').groupBy('event_country_state').count().filter('event_country_state is not null and count >= 10')

# event_geo_location_values_counts = get_category_field_values_counts('geo_location_event', events_df, min_threshold=10)
geo_location_value_cat = events_df.select('geo_location_event').groupBy('geo_location_event').count().filter('geo_location_event is not null and count >= 10')

REG = 0
ctr_udf = F.udf(lambda clicks, views: clicks / float(views + REG), FloatType())

# ### Average CTR by ad_id
ad_id_popularity_df = train_set_df \
    .groupby('ad_id') \
    .agg(F.sum('clicked').alias('clicks'),F.count('*').alias('views')) \
    .withColumn('ctr', F.col('clicks') / F.col('views')) \
    .filter('views > 5').select('ad_id', 'ctr', 'views')

# ### Average CTR by document_id (promoted_content)
document_id_popularity_df = train_set_df \
    .groupby('document_id_promo') \
    .agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views')) \
    .withColumn('ctr', F.col('clicks') / F.col('views')) \
    .filter('views > 5').select('document_id_promo', 'ctr', 'views')

# ### Average CTR by source_id
source_id_popularity_df = train_set_df.select('clicked', 'source_id', 'ad_id') \
    .groupby('source_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views')) \
    .withColumn('ctr', F.col('clicks') / F.col('views')) \
    .filter('views > 10 and source_id is not null').select('source_id', 'ctr')

# ### Average CTR by publisher_id
publisher_popularity_df = train_set_df.select('clicked', 'publisher_id', 'ad_id') \
    .groupby('publisher_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views')) \
    .withColumn('ctr', F.col('clicks') / F.col('views')) \
    .filter('views > 10 and publisher_id is not null').select('publisher_id', 'ctr')

# ### Average CTR by advertiser_id
advertiser_id_popularity_df = train_set_df.select('clicked', 'advertiser_id', 'ad_id') \
    .groupby('advertiser_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views')) \
    .withColumn('ctr', F.col('clicks') / F.col('views')) \
    .filter('views > 10 and advertiser_id is not null').select('advertiser_id', 'ctr')

# ### Average CTR by campaign_id
campaign_id_popularity_df = train_set_df.select('clicked', 'campaign_id', 'ad_id') \
    .groupby('campaign_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views')) \
    .withColumn('ctr', F.col('clicks') / F.col('views')) \
    .filter('views > 10 and campaign_id is not null').select('campaign_id', 'ctr')


documents_categories_schema = StructType(
    [StructField("document_id_cat", IntegerType(), True),
     StructField("category_id", IntegerType(), True),
     StructField("confidence_level_cat", FloatType(), True)]
)

documents_categories_df = spark.read.schema(documents_categories_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_categories.csv") \
    .alias('documents_categories')

documents_topics_schema = StructType(
    [StructField("document_id_top", IntegerType(), True),
     StructField("topic_id", IntegerType(), True),
     StructField("confidence_level_top", FloatType(), True)]
)

documents_topics_df = spark.read.schema(documents_topics_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_topics.csv") \
    .alias('documents_topics')

documents_entities_schema = StructType(
    [StructField("document_id_ent", IntegerType(), True),
     StructField("entity_id", StringType(), True),
     StructField("confidence_level_ent", FloatType(), True)]
)

documents_entities_df = spark.read.schema(documents_entities_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_entities.csv") \
    .alias('documents_entities')

documents_categories_grouped_df = documents_categories_df.groupBy('document_id_cat') \
    .agg(F.collect_list('category_id').alias('category_id_list'),
         F.collect_list('confidence_level_cat').alias('confidence_level_cat_list')) \
    .alias('documents_categories_grouped').cache()

documents_topics_grouped_df = documents_topics_df.groupBy('document_id_top') \
    .agg(F.collect_list('topic_id').alias('topic_id_list'),
         F.collect_list('confidence_level_top').alias('confidence_level_top_list')) \
    .alias('documents_topics_grouped').cache()

documents_entities_grouped_df = documents_entities_df.groupBy('document_id_ent') \
    .agg(F.collect_list('entity_id').alias('entity_id_list'),
         F.collect_list('confidence_level_ent').alias('confidence_level_ent_list')) \
    .alias('documents_entities_grouped').cache()



categories_docs_counts = documents_categories_df.groupBy('category_id').count().rdd.collectAsMap()
categories_docs_counts_broad = sc.broadcast(categories_docs_counts)

topics_docs_counts = documents_topics_df.groupBy('topic_id').count().rdd.collectAsMap()
topics_docs_counts_broad = sc.broadcast(topics_docs_counts)

entities_docs_counts = documents_entities_df.groupBy('entity_id').count().rdd.collectAsMap()
entities_docs_counts_broad = sc.broadcast(entities_docs_counts)


def cosine_similarity_dicts(dict1, dict2):
    dict1_norm = math.sqrt(sum([v ** 2 for v in dict1.values()]))
    dict2_norm = math.sqrt(sum([v ** 2 for v in dict2.values()]))

    sum_common_aspects = 0.0
    for key in dict1:
        if key in dict2:
            sum_common_aspects += dict1[key] * dict2[key]

    return sum_common_aspects / (dict1_norm * dict2_norm)


def cosine_similarity_doc_event_doc_ad_aspects(doc_event_aspect_ids, doc_event_aspects_confidence,
                                               doc_ad_aspect_ids, doc_ad_aspects_confidence,
                                               aspect_docs_counts):
    if doc_event_aspect_ids is None or len(doc_event_aspect_ids) == 0 \
            or doc_ad_aspect_ids is None or len(doc_ad_aspect_ids) == 0:
        return None, None

    doc_event_aspects = dict(zip(doc_event_aspect_ids, doc_event_aspects_confidence))
    doc_event_aspects_tfidf_confid = {}
    for key in doc_event_aspect_ids:
        tf = 1.0
        idf = math.log(math.log(documents_total / float(aspect_docs_counts[key])))
        confidence = doc_event_aspects[key]
        doc_event_aspects_tfidf_confid[key] = tf * idf * confidence

    doc_ad_aspects = dict(zip(doc_ad_aspect_ids, doc_ad_aspects_confidence))
    doc_ad_aspects_tfidf_confid = {}
    for key in doc_ad_aspect_ids:
        tf = 1.0
        idf = math.log(math.log(documents_total / float(aspect_docs_counts[key])))
        confidence = doc_ad_aspects[key]
        doc_ad_aspects_tfidf_confid[key] = tf * idf * confidence

    similarity = cosine_similarity_dicts(doc_event_aspects_tfidf_confid, doc_ad_aspects_tfidf_confid)

    return similarity


#############################################################################################################
def category(df, col, dict_data):
    udf_inter = F.udf(lambda x: float(dict_data.value[x][1]) if x in dict_data.value else None, DoubleType())
    df = df.withColumn(col + '_cat', udf_inter(col))
    return df

def timestamp_delta(df, publish_time, timestamp):
    def timestamp_delta_udf(publish_time, timestamp):
        if timestamp > -1:
            dt_timestamp_event = convert_odd_timestamp(timestamp)
            if publish_time is not None:
                delta_days = (dt_timestamp_event - publish_time).days
                if 0 <= delta_days <= 365 * 10:  # 10 years
                    return float(delta_days)
    udf_inter = F.udf(lambda publish_time, timestamp: timestamp_delta_udf(publish_time, timestamp), DoubleType())
    df = df.withColumn(publish_time + '_delta', udf_inter(publish_time, timestamp))
    return df

# Setting Popularity fields
def get_popularity_score_fn(df, col, dic):
    udf_inter = F.udf(lambda x: float(dic.value[x][0]) if x in dic.value else None, DoubleType())
    df = df.withColumn(col + '_score', udf_inter(col))
    return df

# Setting Doc_event-doc_ad CB Similarity fields
def get_doc_event_doc_ad_cb_similarity_score_fn(df, doc_event_ids, doc_event_levels, doc_ad_ids, doc_ad_levels, cnt):
    udf_inter = F.udf(
        lambda doc_event_ids, doc_event_levels, doc_ad_ids, doc_ad_levels: 
        cosine_similarity_doc_event_doc_ad_aspects(doc_event_ids, doc_event_levels, doc_ad_ids, doc_ad_levels, cnt), DoubleType())
    df = df.withColumn(doc_event_ids + '_sim', udf_inter(doc_event_ids, doc_event_levels, doc_ad_ids, doc_ad_levels))
    return df

def location_codec(df, col, dic):
    udf_inter = F.udf(lambda x: float(dic[x]) if x in dic else dic[LESS_SPECIAL_CAT_VALUE], DoubleType())
    df = df.withColumn(col, udf_inter(col))
    return df


def categorify_join(df, dict_df, column, dict_col):
    dict_df = dict_df.select(column, dict_col)
    df = df.join(dict_df.hint('broadcast'), on=column, how='left') \
        .withColumnRenamed(dict_col, column+'_'+dict_col)
    return df


def days_delta(df, publish_time, timestamp):
    #df = df.withColumn('ts', (F.col(timestamp)+1465876799998)/1000).cast(TimestamType)))
    df = df.withColumn('ts', F.from_unixtime((F.col(timestamp)+1465876799998)/1000)) \
        .withColumn(publish_time + '_delta', F.datediff('ts', publish_time)).drop('ts') \
        .withColumn(publish_time + '_delta', F.when(F.col(publish_time + '_delta') < 0, 0).when(F.col(publish_time + '_delta') > 3650, 0).otherwise(F.col(publish_time + '_delta')))
    return df
#######################################################################################################################

def enrich_df(df):
    df_enriched = df \
        .join(documents_categories_grouped_df.hint('broadcast'),
          on=F.col("document_id_promo") == F.col("documents_categories_grouped.document_id_cat"),
          how='left') \
        .join(documents_topics_grouped_df.hint('broadcast'),
          on=F.col("document_id_promo") == F.col("documents_topics_grouped.document_id_top"),
          how='left') \
        .join(documents_entities_grouped_df.hint('broadcast'),
          on=F.col("document_id_promo") == F.col("documents_entities_grouped.document_id_ent"),
          how='left') \
        .join(documents_categories_grouped_df.hint('broadcast')
          .withColumnRenamed('category_id_list', 'doc_event_category_id_list')
          .withColumnRenamed('confidence_level_cat_list', 'doc_event_confidence_level_cat_list')
          .alias('documents_event_categories_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_categories_grouped.document_id_cat"),
          how='left') \
        .join(documents_topics_grouped_df.hint('broadcast')
          .withColumnRenamed('topic_id_list', 'doc_event_topic_id_list')
          .withColumnRenamed('confidence_level_top_list', 'doc_event_confidence_level_top_list')
          .alias('documents_event_topics_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_topics_grouped.document_id_top"),
          how='left') \
        .join(documents_entities_grouped_df.hint('broadcast')
          .withColumnRenamed('entity_id_list', 'doc_event_entity_id_list')
          .withColumnRenamed('confidence_level_ent_list', 'doc_event_confidence_level_ent_list')
          .alias('documents_event_entities_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_entities_grouped.document_id_ent"),
          how='left') \
        .select('display_id', 'uuid_event', 'event_country', 'event_country_state', 'platform_event',
            'source_id_doc_event', 'publisher_doc_event', 'publish_time_doc_event',
            'publish_time', 'ad_id', 'document_id_promo', 'clicked',
            'geo_location_event', 'advertiser_id', 'publisher_id',
            'campaign_id', 'document_id_event',
            F.coalesce("doc_event_category_id_list", F.array())
            .alias('doc_event_category_id_list'),
            F.coalesce("doc_event_confidence_level_cat_list", F.array())
            .alias('doc_event_confidence_level_cat_list'),
            F.coalesce("doc_event_topic_id_list", F.array())
            .alias('doc_event_topic_id_list'),
            F.coalesce("doc_event_confidence_level_top_list", F.array())
            .alias('doc_event_confidence_level_top_list'),
            F.coalesce("doc_event_entity_id_list", F.array())
            .alias('doc_event_entity_id_list'),
            F.coalesce("doc_event_confidence_level_ent_list", F.array())
            .alias('doc_event_confidence_level_ent_list'),
            F.coalesce("source_id", F.lit(-1)).alias('source_id'),
            F.coalesce("timestamp_event", F.lit(-1)).alias('timestamp_event'),
            F.coalesce("category_id_list", F.array()).alias('category_id_list'),
            F.coalesce("confidence_level_cat_list", F.array())
            .alias('confidence_level_cat_list'),
            F.coalesce("topic_id_list", F.array()).alias('topic_id_list'),
            F.coalesce("confidence_level_top_list", F.array())
            .alias('confidence_level_top_list'),
            F.coalesce("entity_id_list", F.array()).alias('entity_id_list'),
            F.coalesce("confidence_level_ent_list", F.array())
            .alias('confidence_level_ent_list'))
    df_enriched = df_enriched.fillna(-1, subset=['source_id', 'timestamp_event'])
    return df_enriched

def format_number(element, name):
    if name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        return element.cast("int")
    else:
        return element

FEAT_CSV_ORDERED_COLUMNS = ['ad_views', 'campaign_id','doc_views',
                            'doc_event_days_since_published', 'doc_ad_days_since_published',
                            'pop_ad_id', 'pop_document_id', 'pop_publisher_id', 'pop_advertiser_id', 'pop_campain_id',
                            'pop_source_id',
                            'doc_event_doc_ad_sim_categories', 'doc_event_doc_ad_sim_topics',
                            'doc_event_doc_ad_sim_entities', 'ad_advertiser', 'doc_ad_publisher_id',
                            'doc_ad_source_id', 'doc_event_publisher_id', 'doc_event_source_id', 'event_country',
                            'event_country_state', 'event_geo_location', 'event_platform']
feature_vector_labels = ['ad_id_views', 'campaign_id','document_id_promo_views',
                            'publish_time_doc_event_delta', 'publish_time_delta', 
                            'ad_id_ctr', 'document_id_promo_ctr', 'publisher_id_ctr', 
                            'advertiser_id_ctr', 'campaign_id_ctr', 'source_id_ctr', 
                            'doc_event_category_id_list_sim', 'doc_event_topic_id_list_sim',
                            'doc_event_entity_id_list_sim', 
                            'advertiser_id', 'publisher_id', 'source_id', 'publisher_doc_event', 'source_id_doc_event', 
                            'event_country_count', 'event_country_state_count', 'geo_location_event_count', 'platform_event']

def train_fe():
    train_set_enriched_df = enrich_df(train_set_df)
    s_time = time.time()
    train_set_enriched_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_set_enriched_df')
    train_set_enriched_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_set_enriched_df')
    print(f'train_set_enriched_df time: {time.time() - s_time}')

    train_set_features_df = train_set_enriched_df
    train_set_features_df = categorify_join(train_set_features_df, ad_id_popularity_df, 'ad_id', 'views')
    train_set_features_df = categorify_join(train_set_features_df, document_id_popularity_df, 'document_id_promo', 'views')
    train_set_features_df = categorify_join(train_set_features_df, ad_id_popularity_df, 'ad_id', 'ctr')
    train_set_features_df = categorify_join(train_set_features_df, document_id_popularity_df, 'document_id_promo', 'ctr')
    train_set_features_df = categorify_join(train_set_features_df, source_id_popularity_df, 'source_id', 'ctr')
    train_set_features_df = categorify_join(train_set_features_df, publisher_popularity_df, 'publisher_id', 'ctr')
    train_set_features_df = categorify_join(train_set_features_df, advertiser_id_popularity_df, 'advertiser_id', 'ctr')
    train_set_features_df = categorify_join(train_set_features_df, campaign_id_popularity_df, 'campaign_id', 'ctr')
    train_set_features_df = categorify_join(train_set_features_df, country_value_cat, 'event_country', 'count')
    train_set_features_df = categorify_join(train_set_features_df, state_value_cal, 'event_country_state', 'count')
    train_set_features_df = categorify_join(train_set_features_df, geo_location_value_cat, 'geo_location_event', 'count')
    train_set_features_df = train_set_features_df.fillna(0, subset=['event_country_count', 'event_country_state_count', 'geo_location_event_count'])
    train_set_features_df = timestamp_delta(train_set_features_df, 'publish_time', 'timestamp_event')
    train_set_features_df = timestamp_delta(train_set_features_df, 'publish_time_doc_event', 'timestamp_event')
    train_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
        train_set_features_df, 'doc_event_category_id_list', 'doc_event_confidence_level_cat_list', 
        'category_id_list', 'confidence_level_cat_list', categories_docs_counts_broad.value)
    train_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
        train_set_features_df, 'doc_event_topic_id_list', 'doc_event_confidence_level_top_list',
        'topic_id_list', 'confidence_level_top_list', topics_docs_counts_broad.value)
    train_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
        train_set_features_df, 'doc_event_entity_id_list', 'doc_event_confidence_level_ent_list', 
        'entity_id_list', 'confidence_level_ent_list', entities_docs_counts_broad.value)

    train_set_features_df = train_set_features_df.withColumn('platform_event', F.col('platform_event') - 1) \
        .withColumnRenamed('document_id_promo', 'document_id') \
        .withColumnRenamed('clicked', 'label') \
        .withColumn('campaign_id', F.col('campaign_id').cast(DoubleType())) \
        .withColumn('advertiser_id', F.col('advertiser_id').cast(DoubleType())) \
        .withColumn('source_id', F.col('source_id').cast(DoubleType())) \
        .withColumn('publisher_id', F.col('publisher_id').cast(DoubleType())) \
        .withColumn('source_id_doc_event', F.col('source_id_doc_event').cast(DoubleType())) \
        .withColumn('publisher_doc_event', F.col('publisher_doc_event').cast(DoubleType()))

    train_set_features_df = train_set_features_df.fillna(0, subset=feature_vector_labels)

    train_feature_vectors_integral_csv_rdd_df = train_set_features_df.select(
        ['label'] + ['display_id'] + ['ad_id'] + [F.col('document_id').alias('doc_id')] + [F.col('document_id_event').alias('doc_event_id')] + [
            format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
            index, element in enumerate([col(column) for column in feature_vector_labels])]).replace(
        float('nan'), 0)
    train_feature_vectors_integral_csv_rdd_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_feature_vectors_integral_csv_rdd_df')




#####################################################################
def test_fe():
    test_set_enriched_df = enrich_df(valid_set_df)
    s_time = time.time()
    test_set_enriched_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'test_set_enriched_df')
    test_set_enriched_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'test_set_enriched_df')
    print(f'test_set_enriched_df time: {time.time() - s_time}')

    test_set_features_df = test_set_enriched_df
    test_set_features_df = categorify_join(test_set_features_df, ad_id_popularity_df, 'ad_id', 'views')
    test_set_features_df = categorify_join(test_set_features_df, document_id_popularity_df, 'document_id_promo', 'views')
    test_set_features_df = timestamp_delta(test_set_features_df, 'publish_time', 'timestamp_event')
    test_set_features_df = timestamp_delta(test_set_features_df, 'publish_time_doc_event', 'timestamp_event')
    test_set_features_df = categorify_join(test_set_features_df, ad_id_popularity_df, 'ad_id', 'ctr')
    test_set_features_df = categorify_join(test_set_features_df, document_id_popularity_df, 'document_id_promo', 'ctr')
    test_set_features_df = categorify_join(test_set_features_df, source_id_popularity_df, 'source_id', 'ctr')
    test_set_features_df = categorify_join(test_set_features_df, publisher_popularity_df, 'publisher_id', 'ctr')
    test_set_features_df = categorify_join(test_set_features_df, advertiser_id_popularity_df, 'advertiser_id', 'ctr')
    test_set_features_df = categorify_join(test_set_features_df, campaign_id_popularity_df, 'campaign_id', 'ctr')
    test_set_features_df = categorify_join(test_set_features_df, country_value_cat, 'event_country', 'count')
    test_set_features_df = categorify_join(test_set_features_df, state_value_cal, 'event_country_state', 'count')
    test_set_features_df = categorify_join(test_set_features_df, geo_location_value_cat, 'geo_location_event', 'count')
    test_set_features_df = test_set_features_df.fillna(0, subset=['event_country_count', 'event_country_state_count', 'geo_location_event_count'])
    test_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
        test_set_features_df, 'doc_event_category_id_list', 'doc_event_confidence_level_cat_list', 
        'category_id_list', 'confidence_level_cat_list', categories_docs_counts_broad.value)
    test_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
        test_set_features_df, 'doc_event_topic_id_list', 'doc_event_confidence_level_top_list',
        'topic_id_list', 'confidence_level_top_list', topics_docs_counts_broad.value)
    test_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
        test_set_features_df, 'doc_event_entity_id_list', 'doc_event_confidence_level_ent_list', 
        'entity_id_list', 'confidence_level_ent_list', entities_docs_counts_broad.value)
    
    test_set_features_df = test_set_features_df.withColumn('platform_event', F.col('platform_event') - 1) \
        .withColumnRenamed('document_id_promo', 'document_id') \
        .withColumnRenamed('clicked', 'label') \
        .withColumn('campaign_id', F.col('campaign_id').cast(DoubleType())) \
        .withColumn('advertiser_id', F.col('advertiser_id').cast(DoubleType())) \
        .withColumn('source_id', F.col('source_id').cast(DoubleType())) \
        .withColumn('publisher_id', F.col('publisher_id').cast(DoubleType())) \
        .withColumn('source_id_doc_event', F.col('source_id_doc_event').cast(DoubleType())) \
        .withColumn('publisher_doc_event', F.col('publisher_doc_event').cast(DoubleType()))

    test_set_features_df = test_set_features_df.fillna(0, subset=feature_vector_labels)

    test_validation_feature_vectors_integral_csv_rdd_df = test_set_features_df.repartition(40,'display_id').orderBy('display_id').select(
        ['label'] + ['display_id'] + ['ad_id'] + [F.col('document_id').alias('doc_id')] + [F.col('document_id_event').alias('doc_event_id')] + [
            format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
            index, element in enumerate([col(column) for column in feature_vector_labels])]).replace(
        float('nan'), 0)
    test_validation_feature_vectors_integral_csv_rdd_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'test_validation_feature_vectors_integral_csv_rdd_df')

train_fe_thread = threading.Thread(target=train_fe)
train_fe_thread.start()
test_fe_thread = threading.Thread(target=test_fe)
test_fe_thread.start()
train_fe_thread.join()
test_fe_thread.join()

documents_categories_grouped_df.unpersist()
documents_topics_grouped_df.unpersist()
documents_entities_grouped_df.unpersist()
train_feature_vectors_integral_csv_rdd_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_feature_vectors_integral_csv_rdd_df')

test_validation_feature_vectors_integral_csv_rdd_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'test_validation_feature_vectors_integral_csv_rdd_df')

pd.set_option('display.max_columns', 1000)
evaluation = True
evaluation_verbose = False
LOCAL_DATA_TFRECORDS_DIR = "/outbrain/tfrecords-test/rewrite"

TEST_SET_MODE = False

num_train_partitions = 80
num_valid_partitions = 40
# batch_size = PREBATCH_SIZE
batch_size = 2048



CSV_ORDERED_COLUMNS = ['label', 'display_id', 'ad_id', 'doc_id', 'doc_event_id', 'ad_views', 'campaign_id','doc_views',
                       'doc_event_days_since_published', 'doc_ad_days_since_published',
                       'pop_ad_id', 'pop_document_id', 'pop_publisher_id', 'pop_advertiser_id', 'pop_campain_id',
                       'pop_source_id',
                       'doc_event_doc_ad_sim_categories', 'doc_event_doc_ad_sim_topics',
                       'doc_event_doc_ad_sim_entities', 'ad_advertiser', 'doc_ad_publisher_id',
                       'doc_ad_source_id', 'doc_event_publisher_id', 'doc_event_source_id', 'event_country',
                       'event_country_state', 'event_geo_location', 'event_platform',
                       'traffic_source']


def make_spec(output_dir, batch_size=None):
    fixed_shape = [batch_size, 1] if batch_size is not None else []
    spec = {}
    spec[LABEL_COLUMN] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[DISPLAY_ID_COLUMN] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    for name in FLOAT_COLUMNS:
        spec[name] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
    for name in INT_COLUMNS:
        spec[name + '_log_01scaled'] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
    for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        spec[name] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(spec))
    metadata_io.write_metadata(metadata, output_dir)

# write out tfrecords meta
make_spec(LOCAL_DATA_TFRECORDS_DIR + '/transformed_metadata', batch_size=batch_size)


def log2_1p(x):
    return np.log1p(x) / np.log(2.0)


# calculate min and max stats for the given dataframes all in one go
def compute_min_max_logs(df):
    print(str(datetime.datetime.now()) + '\tComputing min and max')
    min_logs = {}
    max_logs = {}
    float_expr = []
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM + INT_COLUMNS:
        float_expr.append(F.min(name))
        float_expr.append(F.max(name))
    floatDf = all_df.agg(*float_expr).collect()
    print(floatDf)
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
        minAgg = floatDf[0]["min(" + name + ")"]
        maxAgg = floatDf[0]["max(" + name + ")"]
        min_logs[name + '_log_01scaled'] = log2_1p(minAgg * 1000)
        max_logs[name + '_log_01scaled'] = log2_1p(maxAgg * 1000)
    for name in INT_COLUMNS:
        minAgg = floatDf[0]["min(" + name + ")"]
        maxAgg = floatDf[0]["max(" + name + ")"]
        min_logs[name + '_log_01scaled'] = log2_1p(minAgg)
        max_logs[name + '_log_01scaled'] = log2_1p(maxAgg)

    return min_logs, max_logs

all_df = test_validation_feature_vectors_integral_csv_rdd_df.union(train_feature_vectors_integral_csv_rdd_df)
min_logs, max_logs = compute_min_max_logs(all_df)


def log_and_norm(df):
    for col in INT_COLUMNS:
        col_name_log = col + '_log_01'
        col_name_norm = col + '_log_01scaled'
        df = df.withColumn(col_name_log, F.log1p(col))
        df = df.withColumn(col_name_norm, (F.col(col_name_log)-min_logs[col_name_norm]) / (max_logs[col_name_norm]-min_logs[col_name_norm]))
        df = df.drop(col_name_log).drop(col)

    df = df.fillna(0, subset=CATEGORICAL_COLUMNS)
    for name, size in HASH_BUCKET_SIZES.items():
        # df = df.withColumn(name, F.udf(lambda x: x % size, IntegerType())(name))
        df = df.withColumn(name, F.col(name) % size).withColumn(name, F.when(F.col(name)<0, F.col(name)+size).otherwise(F.col(name)))
    return df

train_feature_norm = log_and_norm(train_feature_vectors_integral_csv_rdd_df)
test_feature_norm = log_and_norm(test_validation_feature_vectors_integral_csv_rdd_df)

train_feature_norm.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_feature_norm')
train_feature_norm = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_feature_norm')
test_feature_norm.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'test_feature_norm')
test_feature_norm = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'test_feature_norm')

print(f'feature engineering time: {time.time() - clock}')
clock = time.time()
columns = train_feature_norm.columns



def create_tf_example_spark(df):
    result = {}
    result[LABEL_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[LABEL_COLUMN].to_list()))
    result[DISPLAY_ID_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[DISPLAY_ID_COLUMN].to_list()))
    for name in FLOAT_COLUMNS:
        value = df[name].to_list()
        result[name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in INT_COLUMNS:
        nn = name + '_log_01scaled'
        value = df[nn].to_list()
        result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        value = df[name].to_list()
        result[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    tf_example = tf.train.Example(features=tf.train.Features(feature=result))
    return tf_example


max_partition_num = 30


def _transform_to_slices(rdds):
    taskcontext = TaskContext.get()
    partitionid = taskcontext.partitionId()
    csv = pd.DataFrame(list(rdds), columns=columns)
    num_rows = len(csv.index)
    print("working with partition: ", partitionid, max_partition_num, num_rows)
    examples = []
    for start_ind in range(0, num_rows, batch_size if batch_size is not None else 1):  # for each batch
        if start_ind + batch_size - 1 > num_rows:  # if we'd run out of rows
            csv_slice = csv.iloc[start_ind:]
            print("last Example has: ", len(csv_slice), partitionid)
            examples.append((csv_slice, len(csv_slice)))
            return examples
        else:
            csv_slice = csv.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
        examples.append((csv_slice, len(csv_slice)))
    return examples


def _transform_to_tfrecords_from_slices(rdds):
    examples = []
    for slice in rdds:
        if len(slice[0]) != batch_size:
            print("slice size is not correct, dropping: ", len(slice[0]))
        else:
            examples.append(
                (bytearray((create_tf_example_spark(slice[0])).SerializeToString()), None))
    return examples


def _transform_to_tfrecords_from_reslice(rdds):
    examples = []
    all_dataframes = pd.DataFrame([])
    for slice in rdds:
        all_dataframes = all_dataframes.append(slice[0])
    num_rows = len(all_dataframes.index)
    examples = []
    for start_ind in range(0, num_rows, batch_size if batch_size is not None else 1):  # for each batch
        if start_ind + batch_size - 1 > num_rows:  # if we'd run out of rows
            csv_slice = all_dataframes.iloc[start_ind:]
            if TEST_SET_MODE:
                remain_len = batch_size - len(csv_slice)
                (m, n) = divmod(remain_len, len(csv_slice))
                print("remainder: ", len(csv_slice), remain_len, m, n)
                if m:
                    for i in range(m):
                        csv_slice = csv_slice.append(csv_slice)
                csv_slice = csv_slice.append(csv_slice.iloc[:n])
                print("after fill remainder: ", len(csv_slice))
                examples.append(
                    (bytearray((create_tf_example_spark(csv_slice)).SerializeToString()), None))
                return examples
            # drop the remainder
            print("dropping remainder: ", len(csv_slice))
            return examples
        else:
            csv_slice = all_dataframes.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
            examples.append(
                (bytearray((create_tf_example_spark(csv_slice)).SerializeToString()), None))
    return examples

train_output_string = '/train'
eval_output_string = '/eval'

TEST_SET_MODE = False
train_features = train_feature_norm.coalesce(num_train_partitions).rdd.mapPartitions(_transform_to_slices)
cached_train_features = train_features.cache()
train_full = cached_train_features.filter(lambda x: x[1] == batch_size)
# split out slies where we don't have a full batch so that we can reslice them so we only drop mininal rows
train_not_full = cached_train_features.filter(lambda x: x[1] < batch_size)
train_examples_full = train_full.mapPartitions(_transform_to_tfrecords_from_slices)
train_left = train_not_full.coalesce(1).mapPartitions(_transform_to_tfrecords_from_reslice)
all_train = train_examples_full.union(train_left)


valid_features = test_feature_norm.coalesce(num_valid_partitions).rdd.mapPartitions(_transform_to_slices)
cached_valid_features = valid_features.cache()
valid_full = cached_valid_features.filter(lambda x: x[1] == batch_size)
valid_not_full = cached_valid_features.filter(lambda x: x[1] < batch_size)
valid_examples_full = valid_full.mapPartitions(_transform_to_tfrecords_from_slices)
valid_left = valid_not_full.coalesce(1).mapPartitions(_transform_to_tfrecords_from_reslice)
all_valid = valid_examples_full.union(valid_left)

def save(df, path):
    df.saveAsNewAPIHadoopFile(path,
                                "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")

t1 = threading.Thread(target=save, args=(all_train, LOCAL_DATA_TFRECORDS_DIR + train_output_string))
t1.start()
t2 = threading.Thread(target=save, args=(all_valid, LOCAL_DATA_TFRECORDS_DIR + eval_output_string))
t2.start()
t1.join()
t2.join()


print(f'data convert time: {time.time() - clock}')

spark.stop()