import pandas
import pickle
import os.path
import random
from time import time
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from .utils import *
import findspark
findspark.init()


class DataProcessor:

    def __init__(self, spark, reviews_info_df, item_info_df, dir_path):
        self.reviews_info_df = reviews_info_df
        self.item_info_df = item_info_df
        self.spark = spark
        self.dir_path = dir_path
        self.local_prefix = "file://" + dir_path
        self.elapse_time = {}

    def rand_ordinal_n(self, df, n, name='ordinal'):
        return df.withColumn(name, (rand() * n).cast("int"))

    def shuffle_data_by_user(self, df):
        return df.orderBy(df.review_id, df.positive)

    def save_or_cache(self, df, df_name, method=0):
        if method == 0:
            df.write.format('parquet').mode(
                'overwrite').save(self.local_prefix + "/" + df_name)
            return self.spark.read.parquet(self.local_prefix + "/" + df_name)
        else:
            self.df.cache()
            return df

    def get_negative_sample_udf(self, broadcast_data):
        broadcast_movie_id_list = self.spark.sparkContext.broadcast(
            broadcast_data)

        def get_random_id(asin):
            item_list = broadcast_movie_id_list.value
            asin_total_len = len(item_list)
            asin_neg = asin
            while True:
                asin_neg_index = random.randint(0, asin_total_len - 1)
                asin_neg = item_list[asin_neg_index]
                if asin_neg == None or asin_neg == asin:
                    continue
                else:
                    break
            return asin_neg

        return udf(get_random_id, StringType())

    def get_mapping_udf(self, broadcast_data, default="default_cat"):
        broadcast_dict = self.spark.sparkContext.broadcast(
            broadcast_data)

        def get_mapped(asin):
            map_dict = broadcast_dict.value
            if asin in map_dict:
                return map_dict[asin]
            else:
                return default
        return udf(get_mapped, StringType())

    def split_with_random_ratio(self, numerator, denominator, df):
        # all local_test will be split with random 1:9 to local_train_splitByUser and local_test_splitByUser
        df = self.rand_ordinal_n(df, denominator)
        # union_concated_df = reviews_groupby_user_df\
        #                    .join(union_records_df, 'review_id', 'inner')\
        #                    .select('positive',\
        #                            'review_id',\
        #                            'movie_id',\
        #                            'category',\
        #                            'concated_movie_id',\
        #                            'concated_category',\
        #                            'numItemsByUser',\
        #                            'ordinal'
        # split aggregated_labled_df by 1:9
        # self.test_df = reload_union_concated_df.filter(col("ordinal") == 2).drop("ordinal")
        # self.train_df = reload_union_concated_df.filter(col("ordinal") != 2).drop("ordinal")

    def process(self):
        t1 = timer()
        # same as meta_map in process_data.py
        self.meta_map_df = self.item_info_df\
            .groupby("movie_id")\
            .agg(first("category").alias("category"))
        self.meta_map_df = self.save_or_cache(self.meta_map_df, "meta_map")

        # same as user_map in process_data.py
        user_map_df = self.reviews_info_df

        # prepare udfs and window functions
        # asin_list = load_neg_mid(self.dir_path + "/neg_mid_list")
        asin_list = [row['movie_id']
                     for row in self.reviews_info_df.select('movie_id').collect()]
        mid_cat_dict = dict((key, value)
                            for (key, value) in self.meta_map_df.collect())

        get_random_id_udf = self.get_negative_sample_udf(asin_list)
        get_category_udf = self.get_mapping_udf(mid_cat_dict)

        user_window = Window.partitionBy('review_id').orderBy(
            'unix_review_time', 'movie_id')

        # get latest negative records for each user
        last_negative_record_of_user_df = user_map_df\
            .withColumn('uid', row_number().over(Window.partitionBy('review_id').orderBy("unix_review_time")))\
            .withColumn('numItemsByUser', count("movie_id").over(Window.partitionBy('review_id')))\
            .filter(col("uid") == col("numItemsByUser"))\
            .withColumn('movie_id', get_random_id_udf("movie_id"))\
            .withColumn('category', get_category_udf('movie_id'))\
            .select('review_id', 'movie_id', 'category')
        last_negative_record_of_user_df = self.save_or_cache(
            last_negative_record_of_user_df, "aggregated_neg_records")

        # create positive_records_table
        positive_df = user_map_df\
            .join(self.meta_map_df, 'movie_id', 'inner')\
            .select("review_id", "movie_id", "overall", "unix_review_time", "category")

        # generate history records for positives
        last_positive_with_concat_df = positive_df\
            .withColumn('uid', row_number().over(user_window))\
            .withColumn('concated_movie_id', collect_list('movie_id').over(Window.partitionBy('review_id')))\
            .withColumn('concated_category', collect_list('category').over(Window.partitionBy('review_id')))\
            .withColumn('numItemsByUser', size(col("concated_movie_id")))\
            .filter((col("uid") == col("numItemsByUser")) & (col("numItemsByUser") > 2))\
            .withColumn("concated_movie_id", expr("slice(concated_movie_id, 1, numItemsByUser - 1)"))\
            .withColumn("concated_category", expr("slice(concated_category, 1, numItemsByUser - 1)"))\
            .select('review_id', 'movie_id', 'category', 'concated_movie_id', 'concated_category')
        reload_last_positive_with_concat_df = self.save_or_cache(
            last_positive_with_concat_df, "aggregated_records")
        # reload_last_positive_with_concat_df.show()

        # create negative sample records upon above positive records
        last_negative_record_of_user_df = reload_last_positive_with_concat_df\
            .join(last_negative_record_of_user_df, 'review_id', 'inner')\
            .select(
                'review_id',
                lit(0).alias('positive'),
                last_negative_record_of_user_df.movie_id.alias('movie_id'),
                last_negative_record_of_user_df.category.alias('category'),
                'concated_movie_id',
                'concated_category')

        last_positive_record_of_user_df = reload_last_positive_with_concat_df\
            .select('review_id',
                    lit(1).alias('positive'),
                    'movie_id',
                    'category',
                    'concated_movie_id',
                    'concated_category')
        union_records_df = last_negative_record_of_user_df\
            .union(last_positive_record_of_user_df)

        union_concated_df = union_records_df
        reload_union_concated_df = self.save_or_cache(
            union_concated_df, "local_test")
        t2 = timer()
        self.elapse_time["data_process"] = t2 - t1

        self.train_df = reload_union_concated_df

        # saving (using python)

        # build uid_dict, mid_dict and cat_dict
        t3 = timer()
        sorted_uid = [row["review_id"] for row in self.train_df.groupBy('review_id').count().orderBy(
            desc('count')).select('review_id').collect()]
        uid_voc = {}
        uid_voc['A1Y6U82N6TYZPI'] = 0
        uid_voc.update(dict((id, idx) for (id, idx) in zip(
            sorted_uid, range(1, len(sorted_uid) + 1))))
        pickle.dump(uid_voc, open(self.dir_path +
                                  '/uid_voc.pkl', "wb"), protocol=0)
        t4 = timer()
        self.elapse_time['generate_voc'] = t4 - t3

        t3 = timer()
        sorted_mid = [row["movie_id"] for row in self.train_df.withColumn("concated_movie_id", array_union(col("concated_movie_id"), array(col("movie_id"))))
                      .select(explode(col("concated_movie_id")).alias("movie_id"))
                      .groupBy('movie_id')
                      .count()
                      .filter(col('movie_id') != "default_mid").orderBy(desc('count')).select('movie_id').collect()]
        mid_voc = {}
        mid_voc['default_mid'] = 0
        mid_voc.update(dict((id, idx) for (id, idx) in zip(
            sorted_mid, range(1, len(sorted_mid) + 1))))
        pickle.dump(mid_voc, open(self.dir_path +
                                  '/mid_voc.pkl', "wb"), protocol=0)
        t4 = timer()
        self.elapse_time['generate_voc'] += t4 - t3

        t3 = timer()
        sorted_cat = [row["category"] for row in self.train_df.withColumn("concated_category", array_union(col("concated_category"), array(col("category"))))
                      .select(explode(col("concated_category")).alias("category"))
                      .groupBy('category')
                      .count()
                      .filter(col('category') != "default_cat").orderBy(desc('count')).select('category').collect()]
        cat_voc = {}
        cat_voc['default_cat'] = 0
        cat_voc.update(dict((id, idx) for (id, idx) in zip(
            sorted_cat, range(1, len(sorted_cat) + 1))))
        pickle.dump(cat_voc, open(self.dir_path +
                                  '/cat_voc.pkl', "wb"), protocol=0)
        t4 = timer()
        self.elapse_time['generate_voc'] += t4 - t3

        t3 = timer()
        train_dict = [[positive, review_id, movie_id, category, hist_movie_id, hist_category] for (positive, review_id, movie_id, category, hist_movie_id, hist_category) in self.train_df.select('positive',
                                                                                                                                                                                                  'review_id',
                                                                                                                                                                                                  'movie_id',
                                                                                                                                                                                                  'category',
                                                                                                                                                                                                  expr(
                                                                                                                                                                                                      "concat_ws('\x02', concated_movie_id)"),
                                                                                                                                                                                                  expr("concat_ws('\x02', concated_category)")).collect()]
        user_map = {}
        for items in train_dict:
            if items[1] not in user_map:
                user_map[items[1]] = []
            user_map[items[1]].append(items)
        with open(self.dir_path + "/local_train_splitByUser", 'w') as f:
            for user, r in user_map.items():
                positive_sorted = sorted(r, key=lambda x: x[0])
                # print(positive_sorted)
                for items in positive_sorted:
                    print('\t'.join([str(x) for x in items]), file=f)
        t4 = timer()
        self.elapse_time["combine_and_save_negative_positive"] = t4 - t3
