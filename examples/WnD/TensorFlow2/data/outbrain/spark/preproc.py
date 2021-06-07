#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Modifications copyright Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from data.outbrain.features import PREBATCH_SIZE, HASH_BUCKET_SIZES
from data.outbrain.spark.utils.feature_description import LABEL_COLUMN, DISPLAY_ID_COLUMN, CATEGORICAL_COLUMNS, \
    DOC_CATEGORICAL_MULTIVALUED_COLUMNS, BOOL_COLUMNS, INT_COLUMNS, FLOAT_COLUMNS, \
    FLOAT_COLUMNS_LOG_BIN_TRANSFORM, FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM, FLOAT_COLUMNS_NO_TRANSFORM

import os

os.environ["PYSPARK_PYTHON"] = "/root/sw/miniconda3/envs/spark/bin/python"

OUTPUT_BUCKET_FOLDER = "/tmp/spark/preprocessed/"
DATA_BUCKET_FOLDER = "/outbrain/orig/"
SPARK_TEMP_FOLDER = "/tmp/spark/spark-temp/"
TENSORFLOW_HADOOP = "data/outbrain/spark/data/tensorflow-hadoop-1.5.0.jar"


conf = SparkConf().setMaster('spark://sr112:7077').set('spark.executor.memory', '40g').set('spark.driver.memory', '200g').set('spark.executor.cores', '10')
conf.set("spark.jars", TENSORFLOW_HADOOP)
conf.set("spark.sql.files.maxPartitionBytes", 805306368)

# conf.set("spark.sql.adaptive.enabled", True)
# conf.set("spark.sql.autoBroadcastJoinThreshold", "640MB")

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def truncate_day_from_timestamp(ts):
    return int(ts / 1000 / 60 / 60 / 24)

def is_null(value):
    return value is None or len(str(value).strip()) == 0


LESS_SPECIAL_CAT_VALUE = 'less'

# build cat -> cat_count map
def get_category_field_values_counts(field, df, min_threshold=10):
    category_counts = dict(list(filter(lambda x: not is_null(x[0]) and x[1] >= min_threshold,
                                       df.select(field).groupBy(field).count().rdd.map(
                                           lambda x: (x[0], x[1])).collect())))
    # Adding a special value to create a feature for values in this category that are less than min_threshold
    category_counts[LESS_SPECIAL_CAT_VALUE] = -1
    return category_counts

truncate_day_from_timestamp_udf = F.udf(lambda ts: truncate_day_from_timestamp(ts), IntegerType())

extract_country_udf = F.udf(lambda geo: geo.strip()[:2] if geo is not None else '', StringType())

extract_country_state_udf = F.udf(lambda geo: geo.strip()[:5] if geo is not None else '', StringType())

list_len_udf = F.udf(lambda x: len(x) if x is not None else 0, IntegerType())
INT_DEFAULT_NULL_VALUE = -1
int_null_to_minus_one_udf = F.udf(lambda x: x if x is not None else INT_DEFAULT_NULL_VALUE, IntegerType())
int_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(IntegerType()))
float_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(FloatType()))
str_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(StringType()))

def convert_odd_timestamp(timestamp_ms_relative):
    TIMESTAMP_DELTA = 1465876799998
    return datetime.datetime.fromtimestamp((int(timestamp_ms_relative) + TIMESTAMP_DELTA) // 1000)

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
    .withColumn('day_event', truncate_day_from_timestamp_udf('timestamp_event')) \
    .withColumn('event_country', extract_country_udf('geo_location_event')) \
    .withColumn('event_country_state', extract_country_state_udf('geo_location_event')) \
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
source_ids_without_publisher = [row['source_id'] for row in rows_no_pub.collect()]

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
                                  .withColumnRenamed('publish_time', 'publish_time_doc_event'),
                                  on=F.col("document_id_event") == F.col("document_id_doc"), how='left').alias('events').cache()


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

# s_time = time.time()
# valid_set_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'valid_set_df')
# valid_set_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'valid_set_df')
# print(f'valid_set_df time: {time.time() - s_time}')

train_set_df = spark.sql('''
SELECT * FROM clicks_train_joined t
WHERE NOT EXISTS (SELECT display_id FROM validation_display_ids
WHERE display_id = t.display_id)''')

# s_time = time.time()
# train_set_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_set_df')
# train_set_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_set_df')
# print(f'valid_set_df time: {time.time() - s_time}')



################################################################################################################
################################################################################################################
event_country_values_counts = get_category_field_values_counts('event_country', events_df, min_threshold=10)
len(event_country_values_counts)

event_country_state_values_counts = get_category_field_values_counts('event_country_state', events_df, min_threshold=10)
len(event_country_state_values_counts)

event_geo_location_values_counts = get_category_field_values_counts('geo_location_event', events_df, min_threshold=10)
len(event_geo_location_values_counts)


REG = 0
ctr_udf = F.udf(lambda clicks, views: clicks / float(views + REG), FloatType())

# ### Average CTR by ad_id
ad_id_popularity_df = train_set_df.groupby('ad_id').agg(F.sum('clicked').alias('clicks'),
                                                        F.count('*').alias('views')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))


ad_id_popularity = ad_id_popularity_df.filter('views > 5').select('ad_id', 'ctr', 'views') \
    .rdd.map(lambda x: (x['ad_id'], (x['ctr'], x['views']))).collectAsMap()

# ad_id_views = ad_id_popularity_df.filter('views > 5').select('ad_id', 'ctr', 'views') \
#     .rdd.map(lambda x: (x['ad_id'], x['views'])).collectAsMap()

ad_id_popularity_broad = sc.broadcast(ad_id_popularity)
# ad_id_views_broad = sc.broadcast(ad_id_views)

# ### Average CTR by document_id (promoted_content)
document_id_popularity_df = train_set_df \
    .groupby('document_id_promo') \
    .agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
         F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

document_id_popularity = document_id_popularity_df.filter('views > 5') \
    .select('document_id_promo', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['document_id_promo'], (x['ctr'], x['views']))).collectAsMap()

# document_id_views = document_id_popularity_df.filter('views > 5') \
#     .select('document_id_promo', 'ctr', 'views', 'distinct_ad_ids') \
#     .rdd.map(lambda x: (x['document_id_promo'], x['views'])).collectAsMap()

document_id_popularity_broad = sc.broadcast(document_id_popularity)
# document_id_views_broad = sc.broadcast(document_id_views)

# ### Average CTR by source_id
source_id_popularity_df = train_set_df.select('clicked', 'source_id', 'ad_id') \
    .groupby('source_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                              F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

source_id_popularity = source_id_popularity_df \
    .filter('views > 10 and source_id is not null') \
    .select('source_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['source_id'], (x['ctr'], ))) \
    .collectAsMap()

source_id_popularity_broad = sc.broadcast(source_id_popularity)

# ### Average CTR by publisher_id
publisher_popularity_df = train_set_df.select('clicked', 'publisher_id', 'ad_id') \
    .groupby('publisher_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                                 F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

publisher_popularity = publisher_popularity_df \
    .filter('views > 10 and publisher_id is not null') \
    .select('publisher_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['publisher_id'], (x['ctr'], ))) \
    .collectAsMap()

publisher_popularity_broad = sc.broadcast(publisher_popularity)

# ### Average CTR by advertiser_id
advertiser_id_popularity_df = train_set_df.select('clicked', 'advertiser_id', 'ad_id') \
    .groupby('advertiser_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                                  F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

advertiser_id_popularity = advertiser_id_popularity_df \
    .filter('views > 10 and advertiser_id is not null') \
    .select('advertiser_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['advertiser_id'], (x['ctr'], ))) \
    .collectAsMap()

advertiser_id_popularity_broad = sc.broadcast(advertiser_id_popularity)

# ### Average CTR by campaign_id
campaign_id_popularity_df = train_set_df.select('clicked', 'campaign_id', 'ad_id') \
    .groupby('campaign_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                                F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

campaign_id_popularity = campaign_id_popularity_df \
    .filter('views > 10 and campaign_id is not null') \
    .select('campaign_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['campaign_id'], (x['ctr'], ))) \
    .collectAsMap()

campaign_id_popularity_broad = sc.broadcast(campaign_id_popularity)


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

topics_docs_counts = documents_topics_df.groupBy('topic_id').count().rdd.collectAsMap()

entities_docs_counts = documents_entities_df.groupBy('entity_id').count().rdd.collectAsMap()


def get_popularity(an_id, a_dict):
    return a_dict[an_id][0] if an_id in a_dict else None


def get_popularity_score(event_country, ad_id, document_id, source_id,
                         publisher_id, advertiser_id, campaign_id, document_id_event,
                         category_ids_by_doc, cat_confidence_level_by_doc,
                         topic_ids_by_doc, top_confidence_level_by_doc,
                         entity_ids_by_doc, ent_confidence_level_by_doc,
                         output_detailed_list=False):
    probs = []

    avg_ctr = get_popularity(ad_id, ad_id_popularity_broad.value)
    if avg_ctr is not None:
        probs.append(('pop_ad_id', avg_ctr))

    avg_ctr = get_popularity(document_id, document_id_popularity_broad.value)
    if avg_ctr is not None:
        probs.append(('pop_document_id', avg_ctr))


    if source_id != -1:
        avg_ctr = None


        avg_ctr = get_popularity(source_id, source_id_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_source_id', avg_ctr))

    if publisher_id is not None:
        avg_ctr = get_popularity(publisher_id, publisher_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_publisher_id', avg_ctr))

    if advertiser_id is not None:
        avg_ctr = get_popularity(advertiser_id, advertiser_id_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_advertiser_id', avg_ctr))

    if campaign_id is not None:
        avg_ctr = get_popularity(campaign_id, campaign_id_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_campain_id', avg_ctr))

    return probs


def cosine_similarity_dicts(dict1, dict2):
    dict1_norm = math.sqrt(sum([v ** 2 for v in dict1.values()]))
    dict2_norm = math.sqrt(sum([v ** 2 for v in dict2.values()]))

    sum_common_aspects = 0.0
    intersections = 0
    for key in dict1:
        if key in dict2:
            sum_common_aspects += dict1[key] * dict2[key]
            intersections += 1

    return sum_common_aspects / (dict1_norm * dict2_norm), intersections


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

    similarity, intersections = cosine_similarity_dicts(doc_event_aspects_tfidf_confid, doc_ad_aspects_tfidf_confid)

    if intersections > 0:
        # P(A intersect B)_intersections = P(A)^intersections * P(B)^intersections
        random_error = math.pow(len(doc_event_aspect_ids) / float(len(aspect_docs_counts)),
                                intersections) * math.pow(len(doc_ad_aspect_ids) / float(len(aspect_docs_counts)),
                                                          intersections)
    else:
        # P(A not intersect B) = 1 - P(A intersect B)
        random_error = 1 - ((len(doc_event_aspect_ids) / float(len(aspect_docs_counts))) *
                            (len(doc_ad_aspect_ids) / float(len(aspect_docs_counts))))

    confidence = 1.0 - random_error

    return similarity, confidence


def get_doc_event_doc_ad_cb_similarity_score(doc_event_category_ids, doc_event_cat_confidence_levels,
                                             doc_event_topic_ids, doc_event_top_confidence_levels,
                                             doc_event_entity_ids, doc_event_ent_confidence_levels,
                                             doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                             doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                             doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                             output_detailed_list=False):
    # Content-Based
    sims = []

    categories_similarity, cat_sim_confidence = cosine_similarity_doc_event_doc_ad_aspects(
        doc_event_category_ids, doc_event_cat_confidence_levels,
        doc_ad_category_ids, doc_ad_cat_confidence_levels,
        categories_docs_counts)
    if categories_similarity is not None:
        sims.append(('doc_event_doc_ad_sim_categories', categories_similarity, cat_sim_confidence))

    topics_similarity, top_sim_confidence = cosine_similarity_doc_event_doc_ad_aspects(
        doc_event_topic_ids, doc_event_top_confidence_levels,
        doc_ad_topic_ids, doc_ad_top_confidence_levels,
        topics_docs_counts)

    if topics_similarity is not None:
        sims.append(('doc_event_doc_ad_sim_topics', topics_similarity, top_sim_confidence))

    entities_similarity, entity_sim_confid = cosine_similarity_doc_event_doc_ad_aspects(
        doc_event_entity_ids, doc_event_ent_confidence_levels,
        doc_ad_entity_ids, doc_ad_ent_confidence_levels,
        entities_docs_counts)

    if entities_similarity is not None:
        sims.append(('doc_event_doc_ad_sim_entities', entities_similarity, entity_sim_confid))


    return sims

# # Feature Vector export
bool_feature_names = []

int_feature_names = ['ad_views',
                     'campaign_id',
                     'doc_views',
                     'doc_event_days_since_published',
                     'doc_ad_days_since_published',
                     ]

float_feature_names = [
    'pop_ad_id',
    'pop_document_id',
    'pop_publisher_id',
    'pop_advertiser_id',
    'pop_campain_id',
    'pop_source_id',
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_entities',
]

TRAFFIC_SOURCE_FV = 'traffic_source'
EVENT_HOUR_FV = 'event_hour'
EVENT_COUNTRY_FV = 'event_country'
EVENT_COUNTRY_STATE_FV = 'event_country_state'
EVENT_GEO_LOCATION_FV = 'event_geo_location'
EVENT_PLATFORM_FV = 'event_platform'
AD_ADVERTISER_FV = 'ad_advertiser'
DOC_AD_SOURCE_ID_FV = 'doc_ad_source_id'
DOC_AD_PUBLISHER_ID_FV = 'doc_ad_publisher_id'
DOC_EVENT_SOURCE_ID_FV = 'doc_event_source_id'
DOC_EVENT_PUBLISHER_ID_FV = 'doc_event_publisher_id'
DOC_AD_CATEGORY_ID_FV = 'doc_ad_category_id'
DOC_AD_TOPIC_ID_FV = 'doc_ad_topic_id'
DOC_AD_ENTITY_ID_FV = 'doc_ad_entity_id'
DOC_EVENT_CATEGORY_ID_FV = 'doc_event_category_id'
DOC_EVENT_TOPIC_ID_FV = 'doc_event_topic_id'
DOC_EVENT_ENTITY_ID_FV = 'doc_event_entity_id'

# ### Configuring feature vector
category_feature_names_integral = ['ad_advertiser',
                                   'doc_ad_publisher_id',
                                   'doc_ad_source_id',
                                   'doc_event_publisher_id',
                                   'doc_event_source_id',
                                   'event_country',
                                   'event_country_state',
                                   'event_geo_location',
                                   'event_platform',
                                   'traffic_source']

feature_vector_labels_integral = bool_feature_names \
                                 + int_feature_names \
                                 + float_feature_names \
                                 + category_feature_names_integral

feature_vector_labels_integral_dict = dict([(key, idx) for idx, key in enumerate(feature_vector_labels_integral)])


# ### Building feature vectors
def set_feature_vector_cat_value_integral(field_name, field_value, feature_vector):
    if not is_null(field_value):  # and str(field_value) != '-1':
        feature_vector[feature_vector_labels_integral_dict[field_name]] = float(field_value)

def get_ad_feature_vector_integral(
        event_country, event_country_state,
        ad_id, document_id, source_id, doc_ad_publish_time, timestamp_event, platform_event,
        geo_location_event,
        doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
        advertiser_id, publisher_id,
        campaign_id, document_id_event,
        doc_ad_category_ids, doc_ad_cat_confidence_levels,
        doc_ad_topic_ids, doc_ad_top_confidence_levels,
        doc_ad_entity_ids, doc_ad_ent_confidence_levels,
        doc_event_category_ids, doc_event_cat_confidence_levels,
        doc_event_topic_ids, doc_event_top_confidence_levels,
        doc_event_entity_ids, doc_event_ent_confidence_levels):
    try:

        feature_vector = {}
        feature_vector[feature_vector_labels_integral_dict['campaign_id']] = campaign_id
        if ad_id in ad_id_popularity_broad.value:
            feature_vector[feature_vector_labels_integral_dict['ad_views']] = float(
                ad_id_popularity_broad.value[ad_id][1])

        if document_id in document_id_popularity_broad.value:
            feature_vector[feature_vector_labels_integral_dict['doc_views']] = float(
                document_id_popularity_broad.value[document_id][1])

        if timestamp_event > -1:
            dt_timestamp_event = convert_odd_timestamp(timestamp_event)
            if doc_ad_publish_time is not None:
                delta_days = (dt_timestamp_event - doc_ad_publish_time).days
                if 0 <= delta_days <= 365 * 10:  # 10 years
                    feature_vector[feature_vector_labels_integral_dict['doc_ad_days_since_published']] = float(
                        delta_days)

            if doc_event_publish_time is not None:
                delta_days = (dt_timestamp_event - doc_event_publish_time).days
                if 0 <= delta_days <= 365 * 10:  # 10 years
                    feature_vector[feature_vector_labels_integral_dict['doc_event_days_since_published']] = float(
                        delta_days)


        # Setting Popularity fields
        pop_scores = get_popularity_score(event_country, ad_id, document_id, source_id,
                                          publisher_id, advertiser_id, campaign_id, document_id_event,
                                          doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                          doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                          doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                          output_detailed_list=True)

        for score in pop_scores:
            feature_vector[feature_vector_labels_integral_dict[score[0]]] = score[1]


        # Setting Doc_event-doc_ad CB Similarity fields
        doc_event_doc_ad_cb_sim_scores = get_doc_event_doc_ad_cb_similarity_score(
            doc_event_category_ids, doc_event_cat_confidence_levels,
            doc_event_topic_ids, doc_event_top_confidence_levels,
            doc_event_entity_ids, doc_event_ent_confidence_levels,
            doc_ad_category_ids, doc_ad_cat_confidence_levels,
            doc_ad_topic_ids, doc_ad_top_confidence_levels,
            doc_ad_entity_ids, doc_ad_ent_confidence_levels,
            output_detailed_list=True)

        for score in doc_event_doc_ad_cb_sim_scores:
            feature_vector[feature_vector_labels_integral_dict[score[0]]] = score[1]

        # Process code for event_country
        if event_country in event_country_values_counts:
            event_country_code = event_country_values_counts[event_country]
        else:
            event_country_code = event_country_values_counts[LESS_SPECIAL_CAT_VALUE]
        set_feature_vector_cat_value_integral(EVENT_COUNTRY_FV, event_country_code, feature_vector)

        # Process code for event_country_state
        if event_country_state in event_country_state_values_counts:
            event_country_state_code = event_country_state_values_counts[event_country_state]
        else:
            event_country_state_code = event_country_state_values_counts[LESS_SPECIAL_CAT_VALUE]
        set_feature_vector_cat_value_integral(EVENT_COUNTRY_STATE_FV, event_country_state_code, feature_vector)

        # Process code for geo_location_event
        if geo_location_event in event_geo_location_values_counts:
            geo_location_event_code = event_geo_location_values_counts[geo_location_event]
        else:
            geo_location_event_code = event_geo_location_values_counts[LESS_SPECIAL_CAT_VALUE]

        # -1 to traffic_source and platform_event
        if platform_event is not None:
            feature_vector[feature_vector_labels_integral_dict[EVENT_PLATFORM_FV]] = int(platform_event - 1)

        set_feature_vector_cat_value_integral(EVENT_GEO_LOCATION_FV, geo_location_event_code, feature_vector)

        # set_feature_vector_cat_value_integral(EVENT_PLATFORM_FV, platform_event - 1, feature_vector)
        set_feature_vector_cat_value_integral(AD_ADVERTISER_FV, advertiser_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_AD_SOURCE_ID_FV, source_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_AD_PUBLISHER_ID_FV, publisher_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_EVENT_SOURCE_ID_FV, doc_event_source_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_EVENT_PUBLISHER_ID_FV, doc_event_publisher_id, feature_vector)




    except Exception as e:
        raise Exception("[get_ad_feature_vector_integral] ERROR PROCESSING FEATURE VECTOR! Params: {}"
                        .format([event_country, event_country_state,
                                 ad_id, document_id, source_id, doc_ad_publish_time, timestamp_event, platform_event,
                                 geo_location_event,
                                 doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
                                 advertiser_id, publisher_id,
                                 campaign_id, document_id_event,
                                 doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                 doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                 doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                 doc_event_category_ids, doc_event_cat_confidence_levels,
                                 doc_event_topic_ids, doc_event_top_confidence_levels,
                                 doc_event_entity_ids, doc_event_ent_confidence_levels]),
                        e)

    return SparseVector(len(feature_vector_labels_integral_dict), feature_vector)


get_ad_feature_vector_integral_udf = F.udf(
    lambda event_country, event_country_state, ad_id, document_id, source_id,
           doc_ad_publish_time, timestamp_event, platform_event,
           geo_location_event,
           doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
           advertiser_id, publisher_id,
           campaign_id, document_id_event,
           category_ids_by_doc, cat_confidence_level_by_doc,
           topic_ids_by_doc, top_confidence_level_by_doc,
           entity_ids_by_doc, ent_confidence_level_by_doc,
           doc_event_category_id_list, doc_event_confidence_level_cat_list,
           doc_event_topic_id_list, doc_event_confidence_level_top,
           doc_event_entity_id_list, doc_event_confidence_level_ent:
    get_ad_feature_vector_integral(event_country, event_country_state,
                                   ad_id, document_id, source_id, doc_ad_publish_time, timestamp_event,
                                   platform_event,
                                   geo_location_event,
                                   doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
                                   advertiser_id, publisher_id,
                                   campaign_id, document_id_event,
                                   category_ids_by_doc, cat_confidence_level_by_doc,
                                   topic_ids_by_doc, top_confidence_level_by_doc,
                                   entity_ids_by_doc, ent_confidence_level_by_doc,
                                   doc_event_category_id_list, doc_event_confidence_level_cat_list,
                                   doc_event_topic_id_list, doc_event_confidence_level_top,
                                   doc_event_entity_id_list, doc_event_confidence_level_ent),
    VectorUDT())
####################################################################################
########################### split UDF #############################################
##################################################################################
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
        cosine_similarity_doc_event_doc_ad_aspects(doc_event_ids, doc_event_levels, doc_ad_ids, doc_ad_levels, cnt)[0], DoubleType())
    df = df.withColumn(doc_event_ids + '_sim', udf_inter(doc_event_ids, doc_event_levels, doc_ad_ids, doc_ad_levels))
    return df

def location_codec(df, col, dic):
    udf_inter = F.udf(lambda x: float(dic[x]) if x in dic else dic[LESS_SPECIAL_CAT_VALUE], DoubleType())
    df = df.withColumn(col, udf_inter(col))
    return df

def get_feature_vector_integral_fn(
        event_country, event_country_state,
        ad_id_cat, document_id_cat, source_id, doc_ad_publish_time_delta, timestamp_event, platform_event,
        geo_location_event,
        doc_event_source_id, doc_event_publisher_id, doc_event_publish_time_delta,
        advertiser_id, publisher_id,
        campaign_id, document_id_event,
        doc_ad_category_ids, doc_ad_cat_confidence_levels,
        doc_ad_topic_ids, doc_ad_top_confidence_levels,
        doc_ad_entity_ids, doc_ad_ent_confidence_levels,
        doc_event_category_ids, doc_event_cat_confidence_levels,
        doc_event_topic_ids, doc_event_top_confidence_levels,
        doc_event_entity_ids, doc_event_ent_confidence_levels,
        ad_id_score, document_id_score, source_id_score, publisher_id_score, advertiser_id_score, campaign_id_score, 
        doc_event_category_ids_sim, doc_event_topic_ids_sim, doc_event_entity_ids_sim):
    try:

        feature_vector = {}
        feature_vector[feature_vector_labels_integral_dict['campaign_id']] = campaign_id
        if ad_id_cat is not None:
            # feature_vector[feature_vector_labels_integral_dict['ad_views']] = float(ad_id_cat)
            feature_vector[feature_vector_labels_integral_dict['ad_views']] = ad_id_cat
        if document_id_cat is not None:
            # feature_vector[feature_vector_labels_integral_dict['doc_views']] = float(document_id_cat)
            feature_vector[feature_vector_labels_integral_dict['doc_views']] = document_id_cat

        if doc_ad_publish_time_delta is not None:
            # feature_vector[feature_vector_labels_integral_dict['doc_ad_days_since_published']] = float(doc_ad_publish_time_delta)
            feature_vector[feature_vector_labels_integral_dict['doc_ad_days_since_published']] = doc_ad_publish_time_delta
        if doc_event_publish_time_delta is not None:
            # feature_vector[feature_vector_labels_integral_dict['doc_event_days_since_published']] = float(doc_event_publish_time_delta)
            feature_vector[feature_vector_labels_integral_dict['doc_event_days_since_published']] = doc_event_publish_time_delta

        # Setting Popularity fields
        if ad_id_score is not None:
            feature_vector[feature_vector_labels_integral_dict['pop_ad_id']] = ad_id_score
        if document_id_score is not None:
            feature_vector[feature_vector_labels_integral_dict['pop_document_id']] = document_id_score
        if source_id_score is not None:
            feature_vector[feature_vector_labels_integral_dict['pop_source_id']] = source_id_score
        if publisher_id_score is not None:
            feature_vector[feature_vector_labels_integral_dict['pop_publisher_id']] = publisher_id_score
        if advertiser_id_score is not None:
            feature_vector[feature_vector_labels_integral_dict['pop_advertiser_id']] = advertiser_id_score
        if campaign_id_score is not None:
            feature_vector[feature_vector_labels_integral_dict['pop_campain_id']] = campaign_id_score

        # Setting Doc_event-doc_ad CB Similarity fields
        if doc_event_category_ids_sim is not None:
            feature_vector[feature_vector_labels_integral_dict['doc_event_doc_ad_sim_categories']] = doc_event_category_ids_sim
        if doc_event_topic_ids_sim is not None:
            feature_vector[feature_vector_labels_integral_dict['doc_event_doc_ad_sim_topics']] = doc_event_topic_ids_sim
        if doc_event_entity_ids_sim is not None:
            feature_vector[feature_vector_labels_integral_dict['doc_event_doc_ad_sim_entities']] = doc_event_entity_ids_sim

        # Process code for event_country
        # if event_country in event_country_values_counts:
        #     event_country_code = event_country_values_counts[event_country]
        # else:
        #     event_country_code = event_country_values_counts[LESS_SPECIAL_CAT_VALUE]
        set_feature_vector_cat_value_integral(EVENT_COUNTRY_FV, event_country, feature_vector)

        # Process code for event_country_state
        # if event_country_state in event_country_state_values_counts:
        #     event_country_state_code = event_country_state_values_counts[event_country_state]
        # else:
        #     event_country_state_code = event_country_state_values_counts[LESS_SPECIAL_CAT_VALUE]
        set_feature_vector_cat_value_integral(EVENT_COUNTRY_STATE_FV, event_country_state, feature_vector)

        # Process code for geo_location_event
        # if geo_location_event in event_geo_location_values_counts:
        #     geo_location_event_code = event_geo_location_values_counts[geo_location_event]
        # else:
        #     geo_location_event_code = event_geo_location_values_counts[LESS_SPECIAL_CAT_VALUE]

        # -1 to traffic_source and platform_event
        if platform_event is not None:
            feature_vector[feature_vector_labels_integral_dict[EVENT_PLATFORM_FV]] = int(platform_event - 1)

        set_feature_vector_cat_value_integral(EVENT_GEO_LOCATION_FV, geo_location_event, feature_vector)

        # set_feature_vector_cat_value_integral(EVENT_PLATFORM_FV, platform_event - 1, feature_vector)
        set_feature_vector_cat_value_integral(AD_ADVERTISER_FV, advertiser_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_AD_SOURCE_ID_FV, source_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_AD_PUBLISHER_ID_FV, publisher_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_EVENT_SOURCE_ID_FV, doc_event_source_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_EVENT_PUBLISHER_ID_FV, doc_event_publisher_id, feature_vector)

    except Exception as e:
        raise Exception("[get_ad_feature_vector_integral] ERROR PROCESSING FEATURE VECTOR! Params: {}"
                        .format([event_country, event_country_state,
                                 ad_id_cat, document_id_cat, source_id, doc_ad_publish_time_delta, timestamp_event, platform_event,
                                 geo_location_event,
                                 doc_event_source_id, doc_event_publisher_id, doc_event_publish_time_delta,
                                 advertiser_id, publisher_id,
                                 campaign_id, document_id_event,
                                 doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                 doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                 doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                 doc_event_category_ids, doc_event_cat_confidence_levels,
                                 doc_event_topic_ids, doc_event_top_confidence_levels,
                                 doc_event_entity_ids, doc_event_ent_confidence_levels]),
                        e)

    return SparseVector(len(feature_vector_labels_integral_dict), feature_vector)

get_ad_feature_vector_integral_udf_mod = F.udf(
    lambda event_country, event_country_state, ad_id_cat, document_id_cat, source_id,
           doc_ad_publish_time_delta, timestamp_event, platform_event,
           geo_location_event,
           doc_event_source_id, doc_event_publisher_id, doc_event_publish_time_delta,
           advertiser_id, publisher_id,
           campaign_id, document_id_event,
           category_ids_by_doc, cat_confidence_level_by_doc,
           topic_ids_by_doc, top_confidence_level_by_doc,
           entity_ids_by_doc, ent_confidence_level_by_doc,
           doc_event_category_id_list, doc_event_confidence_level_cat_list,
           doc_event_topic_id_list, doc_event_confidence_level_top,
           doc_event_entity_id_list, doc_event_confidence_level_ent, 
           ad_id_score, document_id_score, source_id_score, publisher_id_score, advertiser_id_score, campaign_id_score, 
           doc_event_category_ids_sim, doc_event_topic_ids_sim, doc_event_entity_ids_sim:
    get_feature_vector_integral_fn(event_country, event_country_state,
                                   ad_id_cat, document_id_cat, source_id, doc_ad_publish_time_delta, timestamp_event,
                                   platform_event,
                                   geo_location_event,
                                   doc_event_source_id, doc_event_publisher_id, doc_event_publish_time_delta,
                                   advertiser_id, publisher_id,
                                   campaign_id, document_id_event,
                                   category_ids_by_doc, cat_confidence_level_by_doc,
                                   topic_ids_by_doc, top_confidence_level_by_doc,
                                   entity_ids_by_doc, ent_confidence_level_by_doc,
                                   doc_event_category_id_list, doc_event_confidence_level_cat_list,
                                   doc_event_topic_id_list, doc_event_confidence_level_top,
                                   doc_event_entity_id_list, doc_event_confidence_level_ent,
                                   ad_id_score, document_id_score, source_id_score, publisher_id_score, advertiser_id_score, campaign_id_score, 
                                   doc_event_category_ids_sim, doc_event_topic_ids_sim, doc_event_entity_ids_sim),
    VectorUDT())

#######################################################################################################################

def enrich_df(df):
    df_enriched = df \
        .join(documents_categories_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_categories_grouped.document_id_cat"),
          how='left') \
        .join(documents_topics_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_topics_grouped.document_id_top"),
          how='left') \
        .join(documents_entities_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_entities_grouped.document_id_ent"),
          how='left') \
        .join(documents_categories_grouped_df
          .withColumnRenamed('category_id_list', 'doc_event_category_id_list')
          .withColumnRenamed('confidence_level_cat_list', 'doc_event_confidence_level_cat_list')
          .alias('documents_event_categories_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_categories_grouped.document_id_cat"),
          how='left') \
        .join(documents_topics_grouped_df
          .withColumnRenamed('topic_id_list', 'doc_event_topic_id_list')
          .withColumnRenamed('confidence_level_top_list', 'doc_event_confidence_level_top_list')
          .alias('documents_event_topics_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_topics_grouped.document_id_top"),
          how='left') \
        .join(documents_entities_grouped_df
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
            int_list_null_to_empty_list_udf('doc_event_category_id_list')
            .alias('doc_event_category_id_list'),
            float_list_null_to_empty_list_udf('doc_event_confidence_level_cat_list')
            .alias('doc_event_confidence_level_cat_list'),
            int_list_null_to_empty_list_udf('doc_event_topic_id_list')
            .alias('doc_event_topic_id_list'),
            float_list_null_to_empty_list_udf('doc_event_confidence_level_top_list')
            .alias('doc_event_confidence_level_top_list'),
            str_list_null_to_empty_list_udf('doc_event_entity_id_list')
            .alias('doc_event_entity_id_list'),
            float_list_null_to_empty_list_udf('doc_event_confidence_level_ent_list')
            .alias('doc_event_confidence_level_ent_list'),
            int_null_to_minus_one_udf('source_id').alias('source_id'),
            int_null_to_minus_one_udf('timestamp_event').alias('timestamp_event'),
            int_list_null_to_empty_list_udf('category_id_list').alias('category_id_list'),
            float_list_null_to_empty_list_udf('confidence_level_cat_list')
            .alias('confidence_level_cat_list'),
            int_list_null_to_empty_list_udf('topic_id_list').alias('topic_id_list'),
            float_list_null_to_empty_list_udf('confidence_level_top_list')
            .alias('confidence_level_top_list'),
            str_list_null_to_empty_list_udf('entity_id_list').alias('entity_id_list'),
            float_list_null_to_empty_list_udf('confidence_level_ent_list')
            .alias('confidence_level_ent_list'))
    return df_enriched


def get_feature_vectors(df):
    df_feature_vectors = df \
        .withColumn('feature_vector',
                get_ad_feature_vector_integral_udf_mod(
                    'event_country',
                    'event_country_state',
                    'ad_id_cat',
                    'document_id_promo_cat',
                    'source_id',
                    'publish_time_delta',
                    'timestamp_event',
                    'platform_event',
                    'geo_location_event',
                    'source_id_doc_event',
                    'publisher_doc_event',
                    'publish_time_doc_event_delta',
                    'advertiser_id',
                    'publisher_id',
                    'campaign_id',
                    'document_id_event',
                    'category_id_list',
                    'confidence_level_cat_list',
                    'topic_id_list',
                    'confidence_level_top_list',
                    'entity_id_list',
                    'confidence_level_ent_list',
                    'doc_event_category_id_list',
                    'doc_event_confidence_level_cat_list',
                    'doc_event_topic_id_list',
                    'doc_event_confidence_level_top_list',
                    'doc_event_entity_id_list',
                    'doc_event_confidence_level_ent_list', 
                    'ad_id_score', 'document_id_promo_score', 'source_id_score', 'publisher_id_score', 'advertiser_id_score', 'campaign_id_score', 
                    'doc_event_category_id_list_sim', 'doc_event_topic_id_list_sim', 'doc_event_entity_id_list_sim')) \
        .select(F.col('uuid_event').alias('uuid'), 'display_id', 'ad_id', 'document_id_event',
            F.col('document_id_promo').alias('document_id'), F.col('clicked').alias('label'),
            'feature_vector')
        
    return df_feature_vectors


train_set_enriched_df = enrich_df(train_set_df)
s_time = time.time()
train_set_enriched_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_set_enriched_df')
train_set_enriched_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_set_enriched_df')
print(f'train_set_enriched_df time: {time.time() - s_time}')

test_set_enriched_df = enrich_df(valid_set_df)
s_time = time.time()
test_set_enriched_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'test_set_enriched_df')
test_set_enriched_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'test_set_enriched_df')
print(f'test_set_enriched_df time: {time.time() - s_time}')

documents_categories_grouped_df.unpersist()
documents_topics_grouped_df.unpersist()
documents_entities_grouped_df.unpersist()


train_set_features_df = train_set_enriched_df
train_set_features_df = category(train_set_features_df, 'ad_id', ad_id_popularity_broad)
train_set_features_df = category(train_set_features_df, 'document_id_promo', document_id_popularity_broad)
train_set_features_df = timestamp_delta(train_set_features_df, 'publish_time', 'timestamp_event')
train_set_features_df = timestamp_delta(train_set_features_df, 'publish_time_doc_event', 'timestamp_event')
train_set_features_df = get_popularity_score_fn(train_set_features_df, 'ad_id', ad_id_popularity_broad)
train_set_features_df = get_popularity_score_fn(train_set_features_df, 'document_id_promo', document_id_popularity_broad)
train_set_features_df = get_popularity_score_fn(train_set_features_df, 'source_id', source_id_popularity_broad)
train_set_features_df = get_popularity_score_fn(train_set_features_df, 'publisher_id', publisher_popularity_broad)
train_set_features_df = get_popularity_score_fn(train_set_features_df, 'advertiser_id', advertiser_id_popularity_broad)
train_set_features_df = get_popularity_score_fn(train_set_features_df, 'campaign_id', campaign_id_popularity_broad)
train_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
    train_set_features_df, 'doc_event_category_id_list', 'doc_event_confidence_level_cat_list', 
    'category_id_list', 'confidence_level_cat_list', categories_docs_counts)
train_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
    train_set_features_df, 'doc_event_topic_id_list', 'doc_event_confidence_level_top_list',
    'topic_id_list', 'confidence_level_top_list', topics_docs_counts)
train_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
    train_set_features_df, 'doc_event_entity_id_list', 'doc_event_confidence_level_ent_list', 
    'entity_id_list', 'confidence_level_ent_list', entities_docs_counts)

train_set_features_df = location_codec(train_set_features_df, 'event_country', event_country_values_counts)
train_set_features_df = location_codec(train_set_features_df, 'event_country_state', event_country_state_values_counts)
train_set_features_df = location_codec(train_set_features_df, 'geo_location_event', event_geo_location_values_counts)
#############################################################################
# train_set_features_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_set_features_df')
# train_set_features_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_set_features_df') # 452s / join - 100s
train_set_features_df = train_set_features_df.withColumn('platform_event', F.udf(lambda x: float(x - 1) if x is not None else None, DoubleType())('platform_event')) \
    .withColumn('traffic_source', F.lit(0).cast(DoubleType())) \
    .withColumnRenamed('document_id_promo', 'document_id') \
    .withColumnRenamed('clicked', 'label') \
    .withColumn('campaign_id', F.col('campaign_id').cast(DoubleType())) \
    .withColumn('advertiser_id', F.col('advertiser_id').cast(DoubleType())) \
    .withColumn('source_id', F.col('source_id').cast(DoubleType())) \
    .withColumn('publisher_id', F.col('publisher_id').cast(DoubleType())) \
    .withColumn('source_id_doc_event', F.col('source_id_doc_event').cast(DoubleType())) \
    .withColumn('publisher_doc_event', F.col('publisher_doc_event').cast(DoubleType()))

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
                            'event_country_state', 'event_geo_location', 'event_platform',
                            'traffic_source']
feature_vector_labels = ['ad_id_cat', 'campaign_id','document_id_promo_cat',
                            'publish_time_doc_event_delta', 'publish_time_delta', 
                            'ad_id_score', 'document_id_promo_score', 'publisher_id_score', 
                            'advertiser_id_score', 'campaign_id_score', 'source_id_score', 
                            'doc_event_category_id_list_sim', 'doc_event_topic_id_list_sim',
                            'doc_event_entity_id_list_sim', 
                            'advertiser_id', 'publisher_id', 'source_id', 'publisher_doc_event', 'source_id_doc_event', 
                            'event_country', 'event_country_state', 'geo_location_event', 'platform_event', 
                            'traffic_source']

for column in feature_vector_labels:
    train_set_features_df = train_set_features_df.fillna(0, subset=column)

train_feature_vectors_integral_csv_rdd_df = train_set_features_df.select(
    ['label'] + ['display_id'] + ['ad_id'] + ['document_id'] + ['document_id_event'] + [
        format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
        index, element in enumerate([col(column) for column in feature_vector_labels])]).replace(
    float('nan'), 0)#.cache()
#####################################################################

# train_set_feature_vectors_df = get_feature_vectors(train_set_features_df)
# s_time = time.time()
# train_set_feature_vectors_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_set_feature_vectors_df')
# train_set_feature_vectors_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_set_feature_vectors_df') # 452s / join - 100s
# print(f'train_set_feature_vectors_df time: {time.time() - s_time}')

test_set_features_df = test_set_enriched_df
test_set_features_df = category(test_set_features_df, 'ad_id', ad_id_popularity_broad)
test_set_features_df = category(test_set_features_df, 'document_id_promo', document_id_popularity_broad)
test_set_features_df = timestamp_delta(test_set_features_df, 'publish_time', 'timestamp_event')
test_set_features_df = timestamp_delta(test_set_features_df, 'publish_time_doc_event', 'timestamp_event')
test_set_features_df = get_popularity_score_fn(test_set_features_df, 'ad_id', ad_id_popularity_broad)
test_set_features_df = get_popularity_score_fn(test_set_features_df, 'document_id_promo', document_id_popularity_broad)
test_set_features_df = get_popularity_score_fn(test_set_features_df, 'source_id', source_id_popularity_broad)
test_set_features_df = get_popularity_score_fn(test_set_features_df, 'publisher_id', publisher_popularity_broad)
test_set_features_df = get_popularity_score_fn(test_set_features_df, 'advertiser_id', advertiser_id_popularity_broad)
test_set_features_df = get_popularity_score_fn(test_set_features_df, 'campaign_id', campaign_id_popularity_broad)
test_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
    test_set_features_df, 'doc_event_category_id_list', 'doc_event_confidence_level_cat_list', 
    'category_id_list', 'confidence_level_cat_list', categories_docs_counts)
test_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
    test_set_features_df, 'doc_event_topic_id_list', 'doc_event_confidence_level_top_list',
    'topic_id_list', 'confidence_level_top_list', topics_docs_counts)
test_set_features_df = get_doc_event_doc_ad_cb_similarity_score_fn(
    test_set_features_df, 'doc_event_entity_id_list', 'doc_event_confidence_level_ent_list', 
    'entity_id_list', 'confidence_level_ent_list', entities_docs_counts)

test_set_features_df = location_codec(test_set_features_df, 'event_country', event_country_values_counts)
test_set_features_df = location_codec(test_set_features_df, 'event_country_state', event_country_state_values_counts)
test_set_features_df = location_codec(test_set_features_df, 'geo_location_event', event_geo_location_values_counts)

# test_validation_set_feature_vectors_df = get_feature_vectors(test_set_features_df)
# s_time = time.time()
# test_validation_set_feature_vectors_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'test_validation_set_feature_vectors_df')
# test_validation_set_feature_vectors_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'test_validation_set_feature_vectors_df') # 198s
# print(f'test_validation_set_feature_vectors_df time: {time.time() - s_time}')

#############################################################################
test_set_features_df = test_set_features_df.withColumn('platform_event', F.udf(lambda x: float(x - 1) if x is not None else None, DoubleType())('platform_event')) \
    .withColumn('traffic_source', F.lit(0).cast(DoubleType())) \
    .withColumnRenamed('document_id_promo', 'document_id') \
    .withColumnRenamed('clicked', 'label') \
    .withColumn('campaign_id', F.col('campaign_id').cast(DoubleType())) \
    .withColumn('advertiser_id', F.col('advertiser_id').cast(DoubleType())) \
    .withColumn('source_id', F.col('source_id').cast(DoubleType())) \
    .withColumn('publisher_id', F.col('publisher_id').cast(DoubleType())) \
    .withColumn('source_id_doc_event', F.col('source_id_doc_event').cast(DoubleType())) \
    .withColumn('publisher_doc_event', F.col('publisher_doc_event').cast(DoubleType()))

for column in feature_vector_labels:
    test_set_features_df = test_set_features_df.fillna(0, subset=column)

test_validation_feature_vectors_integral_csv_rdd_df = test_set_features_df.repartition(40,'display_id').orderBy('display_id').select(
    ['label'] + ['display_id'] + ['ad_id'] + ['document_id'] + ['document_id_event'] + [
        format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
        index, element in enumerate([col(column) for column in feature_vector_labels])]).replace(
    float('nan'), 0)#.cache()



pd.set_option('display.max_columns', 1000)
evaluation = True
evaluation_verbose = False
LOCAL_DATA_TFRECORDS_DIR = "/outbrain/tfrecords-test"

TEST_SET_MODE = False

num_train_partitions = 40
num_valid_partitions = 40
batch_size = PREBATCH_SIZE

# # Feature Vector export
bool_feature_names = []

int_feature_names = ['ad_views',
                     'doc_views',
                     'doc_event_days_since_published',
                     'doc_ad_days_since_published',
                     ]

float_feature_names = [
    'pop_ad_id',
    'pop_document_id',
    'pop_publisher_id',
    'pop_advertiser_id',
    'pop_campain_id',
    'pop_source_id',
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_entities',
]

TRAFFIC_SOURCE_FV = 'traffic_source'
EVENT_HOUR_FV = 'event_hour'
EVENT_COUNTRY_FV = 'event_country'
EVENT_COUNTRY_STATE_FV = 'event_country_state'
EVENT_GEO_LOCATION_FV = 'event_geo_location'
EVENT_PLATFORM_FV = 'event_platform'
AD_ADVERTISER_FV = 'ad_advertiser'
DOC_AD_SOURCE_ID_FV = 'doc_ad_source_id'
DOC_AD_PUBLISHER_ID_FV = 'doc_ad_publisher_id'
DOC_EVENT_SOURCE_ID_FV = 'doc_event_source_id'
DOC_EVENT_PUBLISHER_ID_FV = 'doc_event_publisher_id'
DOC_AD_CATEGORY_ID_FV = 'doc_ad_category_id'
DOC_AD_TOPIC_ID_FV = 'doc_ad_topic_id'
DOC_AD_ENTITY_ID_FV = 'doc_ad_entity_id'
DOC_EVENT_CATEGORY_ID_FV = 'doc_event_category_id'
DOC_EVENT_TOPIC_ID_FV = 'doc_event_topic_id'
DOC_EVENT_ENTITY_ID_FV = 'doc_event_entity_id'

# ### Configuring feature vector
category_feature_names_integral = ['ad_advertiser',
                                   'doc_ad_publisher_id',
                                   'doc_ad_source_id',
                                   'doc_event_publisher_id',
                                   'doc_event_source_id',
                                   'event_country',
                                   'event_country_state',
                                   'event_geo_location',
                                   'event_hour',
                                   'event_platform',
                                   'traffic_source']
feature_vector_labels_integral = bool_feature_names \
                                 + int_feature_names \
                                 + float_feature_names \
                                 + category_feature_names_integral

CSV_ORDERED_COLUMNS = ['label', 'display_id', 'ad_id', 'doc_id', 'doc_event_id', 'ad_views', 'campaign_id','doc_views',
                       'doc_event_days_since_published', 'doc_ad_days_since_published',
                       'pop_ad_id', 'pop_document_id', 'pop_publisher_id', 'pop_advertiser_id', 'pop_campain_id',
                       'pop_source_id',
                       'doc_event_doc_ad_sim_categories', 'doc_event_doc_ad_sim_topics',
                       'doc_event_doc_ad_sim_entities', 'ad_advertiser', 'doc_ad_publisher_id',
                       'doc_ad_source_id', 'doc_event_publisher_id', 'doc_event_source_id', 'event_country',
                       'event_country_state', 'event_geo_location', 'event_platform',
                       'traffic_source']

FEAT_CSV_ORDERED_COLUMNS = ['ad_views', 'campaign_id','doc_views',
                            'doc_event_days_since_published', 'doc_ad_days_since_published',
                            'pop_ad_id', 'pop_document_id', 'pop_publisher_id', 'pop_advertiser_id', 'pop_campain_id',
                            'pop_source_id',
                            'doc_event_doc_ad_sim_categories', 'doc_event_doc_ad_sim_topics',
                            'doc_event_doc_ad_sim_entities', 'ad_advertiser', 'doc_ad_publisher_id',
                            'doc_ad_source_id', 'doc_event_publisher_id', 'doc_event_source_id', 'event_country',
                            'event_country_state', 'event_geo_location', 'event_platform',
                            'traffic_source']

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()

    # Important: asNondeterministic requires Spark 2.3 or later
    # It can be safely removed i.e.
    # return udf(to_array_, ArrayType(DoubleType()))(col)
    # but at the cost of decreased performance

    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)


def format_number(element, name):
    if name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        return element.cast("int")
    else:
        return element


# train_feature_vectors_exported_df = train_set_feature_vectors_df
# train_feature_vectors_integral_csv_rdd_df = train_feature_vectors_exported_df.select('label', 'display_id', 'ad_id',
#                                                                                      'document_id', 'document_id_event',
#                                                                                      'feature_vector').withColumn(
#     "featvec", to_array("feature_vector")).select(
#     ['label'] + ['display_id'] + ['ad_id'] + ['document_id'] + ['document_id_event'] + [
#         format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
#         index, element in enumerate([col("featvec")[i] for i in range(len(feature_vector_labels_integral))])]).replace(
#     float('nan'), 0)#.cache()
train_feature_vectors_integral_csv_rdd_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'train_feature_vectors_integral_csv_rdd_df')
train_feature_vectors_integral_csv_rdd_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'train_feature_vectors_integral_csv_rdd_df') #120s


# test_validation_feature_vectors_exported_df = test_validation_set_feature_vectors_df

# test_validation_feature_vectors_exported_df = test_validation_feature_vectors_exported_df.orderBy('display_id')

# test_validation_feature_vectors_integral_csv_rdd_df = test_validation_feature_vectors_exported_df.select(
#     'label', 'display_id', 'ad_id', 'document_id', 'document_id_event', 'feature_vector').withColumn("featvec",
#                                                                                                      to_array(
#                                                                                                          "feature_vector")).select(
#     ['label'] + ['display_id'] + ['ad_id'] + ['document_id'] + ['document_id_event'] + [
#         format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
#         index, element in enumerate([col("featvec")[i] for i in range(len(feature_vector_labels_integral))])]).replace(
#     float('nan'), 0)#.cache()
test_validation_feature_vectors_integral_csv_rdd_df.write.format('parquet').mode('overwrite').save(OUTPUT_BUCKET_FOLDER + 'test_validation_feature_vectors_integral_csv_rdd_df')
test_validation_feature_vectors_integral_csv_rdd_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + 'test_validation_feature_vectors_integral_csv_rdd_df') # 70s


def make_spec(output_dir, batch_size=None):
    fixed_shape = [batch_size, 1] if batch_size is not None else []
    spec = {}
    spec[LABEL_COLUMN] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[DISPLAY_ID_COLUMN] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    for name in BOOL_COLUMNS:
        spec[name] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM + FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM + FLOAT_COLUMNS_NO_TRANSFORM:
        spec[name] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
    for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
        spec[name + '_binned'] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
        spec[name + '_log_01scaled'] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
    for name in INT_COLUMNS:
        spec[name + '_log_01scaled'] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
    for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        spec[name] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    for multi_category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
        shape = fixed_shape[:-1] + [len(DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category])]
        spec[multi_category] = tf.io.FixedLenFeature(shape=shape, dtype=tf.int64)
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
min_logs, max_logs = compute_min_max_logs(all_df) # 1s


train_output_string = '/train'
eval_output_string = '/eval'

path = LOCAL_DATA_TFRECORDS_DIR


def create_tf_example_spark(df, min_logs, max_logs):
    result = {}
    result[LABEL_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[LABEL_COLUMN].to_list()))
    result[DISPLAY_ID_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[DISPLAY_ID_COLUMN].to_list()))
    for name in FLOAT_COLUMNS:
        value = df[name].to_list()
        result[name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
        value = df[name].multiply(10).astype('int64').to_list()
        result[name + '_binned'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
        value_prelim = df[name].multiply(1000).apply(np.log1p).multiply(1. / np.log(2.0))
        value = value_prelim.astype('int64').to_list()
        result[name + '_binned'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        nn = name + '_log_01scaled'
        value = value_prelim.add(-min_logs[nn]).multiply(1. / (max_logs[nn] - min_logs[nn])).to_list()
        result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in INT_COLUMNS:
        value_prelim = df[name].apply(np.log1p).multiply(1. / np.log(2.0))
        value = value_prelim.astype('int64').to_list()
        result[name + '_log_int'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        nn = name + '_log_01scaled'
        value = value_prelim.add(-min_logs[nn]).multiply(1. / (max_logs[nn] - min_logs[nn])).to_list()
        result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        value = df[name].fillna(0).astype('int64').to_list()
        result[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    for multi_category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
        values = []
        for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category]:
            values = values + [df[category].to_numpy()]
        # need to transpose the series so they will be parsed correctly by the FixedLenFeature
        # we can pass in a single series here; they'll be reshaped to [batch_size, num_values]
        # when parsed from the TFRecord
        value = np.stack(values, axis=1).flatten().tolist()
        result[multi_category] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    tf_example = tf.train.Example(features=tf.train.Features(feature=result))
    return tf_example


def hash_bucket(num_buckets):
    return lambda x: x % num_buckets


def _transform_to_tfrecords(rdds):
    csv = pd.DataFrame(list(rdds), columns=CSV_ORDERED_COLUMNS)
    num_rows = len(csv.index)
    examples = []
    for start_ind in range(0, num_rows, batch_size if batch_size is not None else 1):  # for each batch
        if start_ind + batch_size - 1 > num_rows:  # if we'd run out of rows
            csv_slice = csv.iloc[start_ind:]
            # drop the remainder
            print("last Example has: ", len(csv_slice))
            examples.append((create_tf_example_spark(csv_slice, min_logs, max_logs), len(csv_slice)))
            return examples
        else:
            csv_slice = csv.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
        examples.append((create_tf_example_spark(csv_slice, min_logs, max_logs), batch_size))
    return examples


max_partition_num = 30


def _transform_to_slices(rdds):
    taskcontext = TaskContext.get()
    partitionid = taskcontext.partitionId()
    csv = pd.DataFrame(list(rdds), columns=CSV_ORDERED_COLUMNS)
    for name, size in HASH_BUCKET_SIZES.items():
        if name in csv.columns.values:
            csv[name] = csv[name].apply(hash_bucket(size))
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
                (bytearray((create_tf_example_spark(slice[0], min_logs, max_logs)).SerializeToString()), None))
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
                    (bytearray((create_tf_example_spark(csv_slice, min_logs, max_logs)).SerializeToString()), None))
                return examples
            # drop the remainder
            print("dropping remainder: ", len(csv_slice))
            return examples
        else:
            csv_slice = all_dataframes.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
            examples.append(
                (bytearray((create_tf_example_spark(csv_slice, min_logs, max_logs)).SerializeToString()), None))
    return examples


TEST_SET_MODE = False
train_features = train_feature_vectors_integral_csv_rdd_df.coalesce(30).rdd.mapPartitions(_transform_to_slices)
cached_train_features = train_features.cache()
train_full = cached_train_features.filter(lambda x: x[1] == batch_size)
# split out slies where we don't have a full batch so that we can reslice them so we only drop mininal rows
train_not_full = cached_train_features.filter(lambda x: x[1] < batch_size)
train_examples_full = train_full.mapPartitions(_transform_to_tfrecords_from_slices)
train_left = train_not_full.coalesce(1).mapPartitions(_transform_to_tfrecords_from_reslice)
all_train = train_examples_full.union(train_left)

all_train.saveAsNewAPIHadoopFile(LOCAL_DATA_TFRECORDS_DIR + train_output_string,
                                 "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                 keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable") # 150s
train_features.unpersist()


# TEST_SET_MODE = True
valid_features = test_validation_feature_vectors_integral_csv_rdd_df.coalesce(num_valid_partitions).rdd.mapPartitions(_transform_to_slices)
cached_valid_features = valid_features.cache()
valid_full = cached_valid_features.filter(lambda x: x[1] == batch_size)
valid_not_full = cached_valid_features.filter(lambda x: x[1] < batch_size)
valid_examples_full = valid_full.mapPartitions(_transform_to_tfrecords_from_slices)
valid_left = valid_not_full.coalesce(1).mapPartitions(_transform_to_tfrecords_from_reslice)
all_valid = valid_examples_full.union(valid_left)

all_valid.saveAsNewAPIHadoopFile(LOCAL_DATA_TFRECORDS_DIR + eval_output_string,
                                 "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                 keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable") # 79s
valid_features.unpersist()

spark.stop()