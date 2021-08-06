import init

import os
import pandas as pd
import numpy as np
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
from pyspark.sql.types import *
from timeit import default_timer as timer
import logging
from pyrecdp.data_processor import *
from pyrecdp.utils import *

def load_csv(spark, path):
    review_id_field = StructField('reviewer_id', StringType())
    asin_field = StructField('asin', StringType())
    overall_field = StructField('overall', FloatType())
    unix_time_field = StructField('unix_review_time', IntegerType())
    reviews_info_schema = StructType(
        [review_id_field, asin_field, overall_field, unix_time_field])

    category_field = StructField('category', StringType())
    item_info_schema = StructType([asin_field, category_field])

    reviews_info_df = spark.read.schema(reviews_info_schema).option('sep', '\t').csv(path + "/reviews-info")
    item_info_df = spark.read.schema(item_info_schema).option('sep', '\t').csv(path + "/item-info")

    return reviews_info_df, item_info_df


def collapse_by_hist(df, item_info_df, proc, output_name, min_num_hist = 0):
    op_model_merge = ModelMerge([{'col_name': 'asin', 'dict': item_info_df}])
    op_collapse_by_hist = CollapseByHist(['asin', 'category'], by = ['reviewer_id'], orderBy = ['unix_review_time', 'asin'], minNumHist = min_num_hist)
    proc.reset_ops([op_model_merge, op_collapse_by_hist])
    
    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()
    print(f"Merge with category and collapse hist took {t2 - t1} secs")
    return df


def get_dict_for_asin(df, proc):
    dict_dfs = []
    dict_names = ['asin']
    dict_dfs = [{'col_name': name, 'dict': df.select(spk_func.col(name).alias('dict_col'))} for name in dict_names]
    return dict_dfs


def add_negative_sample(df, item_info_df, dict_dfs, proc, output_name):
    # add negative_sample as new row
    op_negative_sample = NegativeSample(['asin'], dict_dfs)
    op_drop_category = DropFeature(['category'])
    op_add_category = ModelMerge([{'col_name': 'asin', 'dict': item_info_df}])
    op_select = SelectFeature(['pos', 'reviewer_id', 'asin', 'category', 'hist_asin', 'hist_category'])
    proc.reset_ops([op_negative_sample, op_drop_category, op_add_category, op_select])
    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()    
    print(f"add_negative_sample took {t2 - t1} secs")
    return df

def save_to_voc(df, proc, cols, default_name, default_v, output_name):
    import pickle
    col_name = ''
    dtypes_list = []
    if isinstance(cols, list):
        col_name = cols[0]
        dtypes_list = df.select(*cols).dtypes
    elif isinstance(cols, str):
        col_name = cols
        dtypes_list = df.select(cols).dtypes
    else:
        raise ValueError("save_to_voc expects cols as String or list of String")

    to_select = []
    if len(dtypes_list) == 1 and 'array' not in dtypes_list[0][1]:
        to_select.append(f.col(dtypes_list[0][0]))
        dict_df = df.select(*to_select)
    else:
        for name, dtype in dtypes_list:
            if 'array' in dtype:
                to_select.append(f.col(name))
            else:
                to_select.append(f.array(f.col(name)))

        dict_df = df.withColumn(col_name, f.array_union(*to_select))
        dict_df = dict_df.select(f.explode(f.col(col_name)).alias(col_name))
    dict_df = dict_df.filter(f"{col_name} is not null").groupBy(col_name).count().orderBy(f.desc('count')).select(col_name)
    collected = [row[col_name] for row in dict_df.collect()]

    voc = {}
    voc[default_name] = default_v
    voc.update(dict((col_id, col_idx) for (col_id, col_idx) in zip(collected, range(1, len(collected) + 1))))
    pickle.dump(voc, open(proc.current_path + f'/{output_name}', "wb"), protocol=0)


def save_to_uid_voc(df, proc):
    # saving (using python)
    # build uid_dict, mid_dict and cat_dict
    t1 = timer()
    save_to_voc(df, proc, ['reviewer_id'], 'A1Y6U82N6TYZPI', 0, 'uid_voc.pkl')
    t2 = timer()
    print(f"save_to_uid_voc took {t2 - t1} secs")
    
def save_to_mid_voc(df, proc):
    t1 = timer()
    save_to_voc(df, proc, ['hist_asin', 'asin'], 'default_mid', 0, 'mid_voc.pkl')
    t2 = timer()
    print(f"save_to_mid_voc took {t2 - t1} secs")
    
    
def save_to_cat_voc(df, proc):
    t1 = timer()
    save_to_voc(df, proc, ['hist_category', 'category'], 'default_cat', 0, 'cat_voc.pkl')
    t2 = timer()
    print(f"save_to_cat_voc took {t2 - t1} secs")
    
    
def save_to_local_train_splitByUser(df, proc):
    t1 = timer()
    dict_df = df.select('pos', 'reviewer_id', 'asin', 'category',
                        f.expr("concat_ws('\x02', hist_asin)"),
                        f.expr("concat_ws('\x02', hist_category)"))
    collected = [[c1, c2, c3, c4, c5, c6] for (c1, c2, c3, c4, c5, c6) in dict_df.collect()]
    user_map = {}
    for items in collected:
        if items[1] not in user_map:
            user_map[items[1]] = []
        user_map[items[1]].append(items)
    with open(proc.current_path + "/local_train_splitByUser", 'w') as fp:
        for user, r in user_map.items():
            positive_sorted = sorted(r, key=lambda x: x[0])
            for items in positive_sorted:
                print('\t'.join([str(x) for x in items]), file=fp)
    t2 = timer()
    print(f"save_to_local_train_splitByUser took {t2 - t1} secs")

def main():
    path_prefix = "file://"
    current_path = "/home/vmagent/app/recdp/examples/python_tests/dien/output/"
    original_folder = "/home/vmagent/app/recdp/examples/python_tests/dien/j2c_test/"

    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    ##### 1. Start spark and initialize data processor #####
    t0 = timer()
    spark = SparkSession.builder.master('local[104]')\
        .appName("dien_data_process")\
        .config("spark.driver.memory", "480G")\
        .config("spark.driver.memoryOverhead", "20G")\
        .config("spark.executor.cores", "104")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')

    # ===============================================
    # basic: Do categorify for all columns
    reviews_info_df, item_info_df = load_csv(spark, path_prefix + original_folder)

    # 1. join records with its category and then collapse history 
    df = reviews_info_df
    dict_dfs = get_dict_for_asin(df, proc)
    df = collapse_by_hist(df, item_info_df, proc, "collapsed", min_num_hist = 2)

    # 2. add negative sample to records
    df = add_negative_sample(df, item_info_df, dict_dfs, proc, "records_with_negative_sample")

    # df = spark.read.parquet(path_prefix + current_path + "records_with_negative_sample")
    save_to_uid_voc(df, proc)
    save_to_mid_voc(df, proc)
    save_to_cat_voc(df, proc)
    save_to_local_train_splitByUser(df, proc)
    t1 = timer()

    print(f"Total process time is {(t1 - t0)} secs")

    ####################################
    
if __name__ == "__main__":
    main()
