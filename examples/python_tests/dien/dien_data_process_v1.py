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


def collapse_by_hist(df, item_info_df, proc, output_name, min_num_hist = 0, max_num_hist = 100):
    op_model_merge = ModelMerge([{'col_name': 'asin', 'dict': item_info_df}])
    op_collapse_by_hist = CollapseByHist(['asin', 'category'], by = ['reviewer_id'], orderBy = ['unix_review_time', 'asin'], minNumHist = min_num_hist, maxNumHist = 100)
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


def add_negative_hist_cols(df, item_info_df, dict_dfs, proc, output_name):
    # add negative_sample as new row
    op_columns_sample = NegativeFeature({'noclk_hist_asin': 'hist_asin'}, dict_dfs, doSplit=True, sep='\x02', negCnt = 5)
    proc.reset_ops([op_columns_sample])
    t1 = timer()

    df = proc.transform(df, output_name)
    t2 = timer()    
    print(f"add_negative_hist_cols took {t2 - t1} secs")
    return df


def load_processed_csv(spark, data_dir):
    label_field = StructField('pos', IntegerType())
    review_id_field = StructField('reviewer_id', StringType())
    asin_field = StructField('asin', StringType())
    category_field = StructField('category', StringType())
    hist_asin_field = StructField('hist_asin', StringType())
    hist_category_field = StructField('hist_category', StringType())
    csv_schema = StructType(
        [label_field, review_id_field, asin_field, category_field, hist_asin_field, hist_category_field])

    return spark.read.schema(csv_schema).option('sep', '\t').csv(data_dir)


def categorify_dien_data(df, user_df, asin_df, cat_df, asin_cat_df, proc, output_name):
    df = df.select('pos', 'reviewer_id', 'asin', 'category', 'hist_asin', 'hist_category',
                   'noclk_hist_asin', f.expr('noclk_hist_asin as noclk_hist_category'))

    dict_dfs = []
    dict_dfs.append({'col_name': 'reviewer_id', 'dict': user_df})
    dict_dfs.append({'col_name': 'asin', 'dict': asin_df})
    dict_dfs.append({'col_name': 'category', 'dict': cat_df})
    dict_dfs.append({'col_name': 'hist_asin', 'dict': asin_df})
    dict_dfs.append({'col_name': 'hist_category', 'dict': cat_df})
    dict_dfs.append({'col_name': 'noclk_hist_asin', 'dict': asin_df})
    dict_dfs.append({'col_name': 'noclk_hist_category', 'dict': asin_cat_df})

    op_categorify = Categorify(['reviewer_id', 'asin', 'category'], dict_dfs=dict_dfs)
    op_categorify_2 = Categorify(['hist_asin', 'hist_category'], dict_dfs=dict_dfs, doSplit=True, sep='\x02')
    op_categorify_3 = Categorify(['noclk_hist_asin', 'noclk_hist_category'], dict_dfs=dict_dfs, doSplit=True, multiLevelSplit=True, multiLevelSep=['|'])
    op_fillna = FillNA(['asin', 'category'], 0)
    proc.reset_ops([op_categorify, op_categorify_2, op_categorify_3, op_fillna])

    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()
    print(f"categorify took {t2 - t1} secs")
    return df


def save_to_voc(df, proc, fmt, cols, default_name, default_v, output_name):
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
    if fmt == 'pkl':
        dict_df = dict_df.filter(f"{col_name} is not null").groupBy(col_name).count().orderBy(f.desc('count')).select(col_name)
        collected = [row[col_name] for row in dict_df.collect()]

        voc = {}
        voc[default_name] = default_v
        voc.update(dict((col_id, col_idx) for (col_id, col_idx) in zip(collected, range(1, len(collected) + 1))))
        pickle.dump(voc, open(proc.current_path + f'/{output_name}.pkl', "wb"), protocol=0)
    else:
        op_gen_dict = GenerateDictionary([f"{col_name}"])
        proc.reset_ops([op_gen_dict])
        dict_dfs = proc.generate_dicts(dict_df)
        dict_df = dict_dfs[0]['dict']
        if fmt == 'adv_pkl':
            voc_count = dict_df.count()
            pickle.dump(voc_count, open(proc.current_path + f'/{output_name}.pkl', "wb"), protocol=0)

    return dict_df


def save_to_uid_voc(df, proc, fmt = 'pkl'):
    # saving (using python)
    # build uid_dict, mid_dict and cat_dict
    t1 = timer()
    dict_df = save_to_voc(df, proc, fmt, ['reviewer_id'], 'A1Y6U82N6TYZPI', 0, 'uid_voc')
    t2 = timer()
    print(f"save_to_uid_voc took {t2 - t1} secs")
    return dict_df
    

def save_to_mid_voc(df, proc, fmt = 'pkl'):
    t1 = timer()
    dict_df = save_to_voc(df, proc, fmt, ['hist_asin', 'asin'], 'default_mid', 0, 'mid_voc')
    t2 = timer()
    print(f"save_to_mid_voc took {t2 - t1} secs")
    return dict_df
    
    
def save_to_cat_voc(df, proc, fmt = 'pkl'):
    t1 = timer()
    dict_df = save_to_voc(df, proc, fmt, ['hist_category', 'category'], 'default_cat', 0, 'cat_voc')
    t2 = timer()
    print(f"save_to_cat_voc took {t2 - t1} secs")
    return dict_df
    
    
def save_to_local_splitByUser(df, proc, output_name):
    hint = 'byRename'
    #hint = 'byRewrite'
    t1 = timer()
    if 'noclk_hist_asin' in df.columns:
        dict_df = df.select('pos', 'reviewer_id', 'asin', 'category',
                            f.expr("concat_ws('\x02', hist_asin)"),
                            f.expr("concat_ws('\x02', hist_category)"),
                            f.expr("concat_ws('\x02', noclk_hist_asin)"),
                            f.expr("concat_ws('\x02', noclk_hist_category)"))
        if hint == 'byRename':
            dict_df.repartition(1).write.format("csv").option('sep', '\t').mode("overwrite").save(f"{proc.path_prefix}{proc.current_path}/{output_name}-spark")
            result_rename_or_convert(proc.current_path, output_name)
            t2 = timer()
            print(f"save_to_local_splitByUser took {t2 - t1} secs")
            return dict_df
        else:
            collected = [[c1, c2, c3, c4, c5, c6, c7, c8] for (c1, c2, c3, c4, c5, c6, c7, c8) in dict_df.collect()]
    else:
        dict_df = df.select('pos', 'reviewer_id', 'asin', 'category',
                            f.expr("concat_ws('\x02', hist_asin)"),
                            f.expr("concat_ws('\x02', hist_category)"))
        if hint == 'byRename':
            dict_df.repartition(1).write.format("csv").option('sep', '\t').mode("overwrite").save(f"{proc.path_prefix}{proc.current_path}/{output_name}-spark")
            result_rename_or_convert(proc.current_path, output_name)
            t2 = timer()
            print(f"save_to_local_splitByUser took {t2 - t1} secs")
            return dict_df
        else:
            collected = [[c1, c2, c3, c4, c5, c6] for (c1, c2, c3, c4, c5, c6) in dict_df.collect()]

    user_map = {}
    for items in collected:
        if items[1] not in user_map:
            user_map[items[1]] = []
        user_map[items[1]].append(items)
    with open(proc.current_path + f"/{output_name}", 'w') as fp:
        for user, r in user_map.items():
            positive_sorted = sorted(r, key=lambda x: x[0])
            for items in positive_sorted:
                print('\t'.join([str(x) for x in items]), file=fp)

    t2 = timer()
    print(f"save_to_local_splitByUser took {t2 - t1} secs")
    return dict_df


def result_rename_or_convert(fpath, output_name):
    source_path_dict = list_dir(fpath)
    fix = "-spark"
    tgt_path = f"{fpath}/{output_name}"
    print(f"result renamed from {source_path_dict[output_name + fix]} to {tgt_path}")
    os.rename(source_path_dict[output_name + fix], tgt_path)
    shutil.rmtree(source_path_dict[output_name + fix], ignore_errors=True)


def main(option = '--basic'):
    if option == '--basic':
        fmt = 'pkl'
    elif option == '--advanced':
        fmt = 'adv_pkl'
    else:
        raise NotImplementedError(f'{option} is not recognized.')

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
    spark.sparkContext.setLogLevel("ERROR")

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')

    # ===============================================
    # basic: Do categorify for all columns
    reviews_info_df, item_info_df = load_csv(spark, path_prefix + original_folder)

    # 1. join records with its category and then collapse history 
    df = reviews_info_df
    dict_dfs = get_dict_for_asin(df, proc)
    df = collapse_by_hist(df, item_info_df, proc, "collapsed", min_num_hist = 2, max_num_hist = 100)

    # 2. add negative sample to records
    df = add_negative_sample(df, item_info_df, dict_dfs, proc, "records_with_negative_sample")


    if fmt == 'adv_pkl':
        dict_dfs[0]['col_name'] = 'hist_asin'
        df = add_negative_hist_cols(df, item_info_df, dict_dfs, proc, "records_with_negative_hists")

    uid_dict_df = save_to_uid_voc(df, proc, fmt = fmt)
    mid_dict_df = save_to_mid_voc(df, proc, fmt = fmt)
    cat_dict_df = save_to_cat_voc(df, proc, fmt = fmt)
    if fmt == 'adv_pkl':
        # for create noclk_hist_category, we should create a dict to mapping from asin to its cat_id
        asin_df = item_info_df.withColumnRenamed('asin', 'dict_col')
        cat_dict_for_merge_df = cat_dict_df.withColumnRenamed('dict_col', 'category')
        op_asin_to_cat_id = ModelMerge([{'col_name': 'category', 'dict': cat_dict_for_merge_df}])
        op_select = SelectFeature(['dict_col', 'dict_col_id', 'count'])
        proc.reset_ops([op_asin_to_cat_id, op_select])
        asin_cat_df = proc.apply(asin_df)

        df = categorify_dien_data(df, uid_dict_df, mid_dict_df, cat_dict_df, asin_cat_df, proc, "local_train_splitByUser.parquet")
        df = save_to_local_splitByUser(df, proc, 'local_train_splitByUser')

        test_df = load_processed_csv(spark, path_prefix + original_folder + "/local_test_splitByUser")
        test_df = add_negative_hist_cols(test_df, item_info_df, dict_dfs, proc, "test_records_with_negative_hists")
        test_df = categorify_dien_data(test_df, uid_dict_df, mid_dict_df, cat_dict_df, asin_cat_df, proc, "local_test_splitByUser.parquet")
        test_df = save_to_local_splitByUser(test_df, proc, 'local_test_splitByUser')
    else:
        train_df = save_to_local_splitByUser(df, proc, 'local_train_splitByUser')
    t1 = timer()

    print(f"Total process time is {(t1 - t0)} secs")

    ####################################
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
