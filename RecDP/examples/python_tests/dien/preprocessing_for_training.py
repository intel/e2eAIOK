import init

import os
import numpy as np
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
from pyspark.sql.types import *
from timeit import default_timer as timer
import logging
from pyrecdp.data_processor import *
from pyrecdp.utils import *

NUM_INSTS = 1

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

def process_reviews_2(spark, review_file, proc, name):
    def process_single_review(row):
        obj = eval(row)
        return [obj['reviewerID'], obj['asin'], obj['overall'], obj['unixReviewTime']]

    records_df = spark.read.text(review_file)
    proc_udf = f.udf(lambda x: process_single_review(x), ArrayType(StringType()))
    records_df = records_df.withColumn('parsed', proc_udf(f.col('value')))
    records_df = records_df.withColumn('reviewer_id', f.col('parsed').getItem(0))
    records_df = records_df.withColumn('asin', f.col('parsed').getItem(1))
    records_df = records_df.withColumn('overall', f.col('parsed').getItem(2))
    records_df = records_df.withColumn('unix_review_time', f.col('parsed').getItem(3))
    records_df = records_df.drop('value').drop('parsed')
    records_df = proc.transform(records_df, name)
    return records_df


def process_reviews(spark, review_file, proc, name):
    records_df = spark.read.json(review_file)
    records_df = records_df.select('reviewerID', 'asin', 'overall', 'unixReviewTime')
    records_df = records_df.withColumnRenamed('reviewerID', 'reviewer_id')
    records_df = records_df.withColumnRenamed('unixReviewTime', 'unix_review_time')
    records_df = proc.transform(records_df, name)
    return records_df


def process_meta(spark, meta_file, proc, name):
    def process_single_meta(row):
        obj = eval(row)
        cat = obj['categories'][0][-1]
        return [obj['asin'], cat]

    meta_df = spark.read.text(meta_file)
    proc_meta_udf = f.udf(lambda x: process_single_meta(x), ArrayType(StringType()))
    meta_df = meta_df.withColumn('asin', proc_meta_udf(f.col('value')).getItem(0))
    meta_df = meta_df.withColumn('category', proc_meta_udf(f.col('value')).getItem(1))
    meta_df = meta_df.drop('value')
    meta_df = proc.transform(meta_df, name)
    return meta_df

def merge_item_to_reviews(df, item_info_df, proc):
    # merge with item-info and reviews-info firstly
    op_asin_to_cat_id = ModelMerge([{'col_name': 'asin', 'dict': item_info_df}])
    proc.reset_ops([op_asin_to_cat_id])
    df = proc.apply(df)
    return df


def collapse_by_hist(df, proc, output_name, min_num_hist = 0, max_num_hist = 100):
    op_collapse_by_hist = CollapseByHist(['asin', 'category'], by = ['reviewer_id'], orderBy = ['unix_review_time', 'asin'], minNumHist = min_num_hist, maxNumHist = 100)
    proc.reset_ops([op_collapse_by_hist])
    
    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()
    print(f"collapse hist took {t2 - t1} secs")
    return df


def get_dict_for_asin(df, proc):
    dict_dfs = []
    dict_names = ['asin']
    dict_dfs = [{'col_name': name, 'dict': df.select(spk_func.col(name).alias('dict_col'))} for name in dict_names]
    return dict_dfs


def add_negative_sample(df, dict_dfs, mid_cat_df, proc, output_name):
    # add negative_sample as new row
    op_negative_sample = NegativeSample(['asin'], dict_dfs)
    # negative row category is not correct
    mid_cat_df = mid_cat_df.withColumnRenamed('dict_col', 'asin').withColumnRenamed('dict_col_id', 'category')
    op_drop_category = DropFeature(['category'])
    op_add_category = ModelMerge([{'col_name': 'asin', 'dict': mid_cat_df}])

    op_select = SelectFeature(['pos', 'reviewer_id', 'asin', 'category', 'hist_asin', 'hist_category'])
    proc.reset_ops([op_negative_sample, op_drop_category, op_add_category, op_select])
    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()    
    print(f"add_negative_sample took {t2 - t1} secs")
    return df


def add_negative_hist_cols(df, dict_dfs, mid_dict_df, mid_cat_df, proc, output_name):
    # since we already categorified asin, we need to create a number_asin dict
    new_dict_dfs = dict_dfs
    new_dict_dfs[0]['col_name'] = 'hist_asin'
    op_columns_sample = NegativeFeature({'noclk_hist_asin': 'hist_asin'}, new_dict_dfs, doSplit=True, sep='\x02', negCnt = 5)
    proc.reset_ops([op_columns_sample])
    df = proc.apply(df)

    # add one col in df to create noclk_hist_category
    df = df.withColumn('noclk_hist_category', f.col('noclk_hist_asin'))

    new_dict_dfs = []
    new_dict_dfs.append({'col_name': 'noclk_hist_category', 'dict': mid_cat_df})
    op_categorify = Categorify(['noclk_hist_category'], dict_dfs=new_dict_dfs, doSplit=True, multiLevelSplit=True, multiLevelSep=['|'])
    proc.reset_ops([op_categorify])
    df = proc.apply(df)

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


def categorify_dien_data(df, user_df, asin_df, cat_df, proc, output_name):
    dict_dfs = []
    dict_dfs.append({'col_name': 'reviewer_id', 'dict': user_df})
    dict_dfs.append({'col_name': 'asin', 'dict': asin_df})
    dict_dfs.append({'col_name': 'category', 'dict': cat_df})

    op_categorify = Categorify(['reviewer_id', 'asin', 'category'], dict_dfs=dict_dfs)
    op_fillna = FillNA(['asin', 'category'], 0)
    proc.reset_ops([op_categorify, op_fillna])

    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()
    print(f"categorify took {t2 - t1} secs")
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

    op_gen_dict = GenerateDictionary([f"{col_name}"])
    proc.reset_ops([op_gen_dict])
    dict_dfs = proc.generate_dicts(dict_df)
    dict_df = dict_dfs[0]['dict']
    voc = {}
    voc.update(dict((row['dict_col'], row['dict_col_id']) for row in dict_df.collect()))
    voc_count = dict_df.count()
    pickle.dump(voc_count, open("/home/xxx/dien/" + f'/{output_name}.pkl', "wb"), protocol=0)
    pickle.dump(voc, open("/home/xxx/dien/" + f'/test_{output_name}.pkl', "wb"), protocol=0)

    return dict_df


def save_to_uid_voc(df, proc):
    # saving (using python)
    # build uid_dict, mid_dict and cat_dict
    t1 = timer()
    dict_df = save_to_voc(df, proc, ['reviewer_id'], 'A1Y6U82N6TYZPI', 0, 'uid_voc')
    t2 = timer()
    print(f"save_to_uid_voc took {t2 - t1} secs")
    return dict_df
    

def save_to_mid_voc(df, proc):
    t1 = timer()
    dict_df = save_to_voc(df, proc, ['asin'], 'default_mid', 0, 'mid_voc')
    t2 = timer()
    print(f"save_to_mid_voc took {t2 - t1} secs")
    return dict_df
    
    
def save_to_cat_voc(df, proc):
    t1 = timer()
    dict_df = save_to_voc(df, proc, ['category'], 'default_cat', 0, 'cat_voc')
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
            dict_df.repartition(NUM_INSTS).write.format("csv").option('sep', '\t').mode("overwrite").save(f"file:///home/vmagent/app/recdp/examples/python_tests/{proc.current_path}/{output_name}-spark")
            result_rename_or_convert("/home/vmagent/app/recdp/examples/python_tests/" + proc.current_path, output_name)
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
            dict_df.repartition(NUM_INSTS).write.format("csv").option('sep', '\t').mode("overwrite").save(f"file:///home/vmagent/app/recdp/examples/python_tests/{proc.current_path}/{output_name}-spark")
            result_rename_or_convert("/home/vmagent/app/recdp/examples/python_tests/" + proc.current_path, output_name)
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
    with open("/home/vmagent/app/recdp/examples/python_tests/" + proc.current_path + f"/{output_name}", 'w') as fp:
        for user, r in user_map.items():
            positive_sorted = sorted(r, key=lambda x: x[0])
            for items in positive_sorted:
                print('\t'.join([str(x) for x in items]), file=fp)

    t2 = timer()
    print(f"save_to_local_splitByUser took {t2 - t1} secs")
    return dict_df


def result_rename_or_convert(fpath, output_name):
    source_path_dict = list_dir(fpath, False)
    fix = "-spark"
    tgt_path = f"/home/xxx/dien/{output_name}"
    idx = 0
    if len(source_path_dict[output_name + fix]) == 1:
        file_name = source_path_dict[output_name + fix][0]
        shutil.copy(file_name, f"{tgt_path}")
    else:
        for file_name in source_path_dict[output_name + fix]:
            #print(f"result renamed from {file_name} to {tgt_path}_{idx}")
            shutil.copy(file_name, f"{tgt_path}_{idx}")
            shutil.rmtree(file_name, ignore_errors=True)
            idx += 1


def main(option = '--advanced'):
    fmt = 'adv_pkl'

    path_prefix = "file://"
    current_path = "/home/vmagent/app/recdp/examples/python_tests/dien/output/"
    original_folder = "/home/vmagent/app/recdp/examples/python_tests/dien/"

    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    ##### 1. Start spark and initialize data processor #####
    t0 = timer()
    spark = SparkSession.builder.master('spark://dr8s30:7077')\
        .appName("dien_data_process")\
        .config("spark.driver.memory", "20G")\
        .config("spark.driver.memoryOverhead", "10G")\
        .config("spark.executor.instances", "4")\
        .config("spark.executor.cores", "32")\
        .config("spark.executor.memory", "100G")\
        .config("spark.executor.memoryOverhead", "20G")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .config("spark.executor.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='standalone')
    t1 = timer()
    print(f"start spark process took {(t1 - t0)} secs")

    # ===============================================
    # Data ingestion
    t0 = timer()
    reviews_info_df = process_reviews(spark, "%s/%s/raw_data/reviews_Books.json" % (path_prefix, original_folder), proc, "reviews-info")
    #reviews_info_df.repartition(1).write.format("csv").option('sep', '\t').mode("overwrite").save("%s/%s/j2c_test/reviews-info-spark" % (path_prefix, original_folder))
    t1 = timer()
    print(f"parse reviews-info with spark took {(t1 - t0)} secs")

    t0 = timer()
    item_info_df = process_meta(spark, '%s/%s/raw_data/meta_Books.json' % (path_prefix, original_folder), proc, "item-info")
    #item_info_df.repartition(1).write.format("csv").option('sep', '\t').mode("overwrite").save("%s/%s/j2c_test/item-info-spark" % (path_prefix, original_folder))
    t1 = timer()
    print(f"parse item-info with spark took {(t1 - t0)} secs")

    # ===============================================
    # basic: Do categorify for all columns
    # reviews_info_df, item_info_df = load_csv(spark, path_prefix + original_folder)

    # 1. join records with its category and then collapse history 
    df = reviews_info_df
    # asin could come from reviews or item, so we do a union here
    #asin_df = df.select('asin').union(item_info_df.select('asin'))
    asin_df = df

    dict_dfs = get_dict_for_asin(df, proc)
    uid_dict_df = save_to_uid_voc(df, proc)
    mid_dict_df = save_to_mid_voc(asin_df, proc)
    cat_dict_df = save_to_cat_voc(item_info_df, proc)

    df = merge_item_to_reviews(df, item_info_df, proc)

    if fmt == 'adv_pkl':
        mid_dict_for_merge_df = mid_dict_df.withColumnRenamed('dict_col', 'asin').withColumnRenamed('dict_col_id', 'asin_id')
        cat_dict_for_merge_df = cat_dict_df.withColumnRenamed('dict_col', 'category').withColumnRenamed('dict_col_id', 'cat_id')

        # generate categorified dict_dfs
        op_asin_to_id = ModelMerge([{'col_name': 'dict_col', 'dict': mid_dict_for_merge_df.withColumnRenamed('asin', 'dict_col')}])
        proc.reset_ops([op_asin_to_id])
        asin_id_dict_df = proc.apply(dict_dfs[0]['dict'])
        asin_id_dict_df = asin_id_dict_df.select('asin_id').withColumnRenamed('asin_id', 'dict_col')
        categorified_dict_dfs = []
        categorified_dict_dfs.append({'col_name': 'asin', 'dict': asin_id_dict_df})

        # generate item id to cat id map
        op_asin_id_to_cat = ModelMerge([{'col_name': 'asin', 'dict': item_info_df}])
        proc.reset_ops([op_asin_id_to_cat])
        asin_cat_df = proc.apply(mid_dict_for_merge_df)
        asin_cat_df.printSchema()

        op_asin_id_to_cat_id = ModelMerge([{'col_name': 'category', 'dict': cat_dict_for_merge_df}])
        proc.reset_ops([op_asin_id_to_cat_id])
        asin_cat_df = proc.apply(asin_cat_df)
        asin_cat_df.printSchema()
        asin_cat_df = asin_cat_df.select('asin_id', 'cat_id')
        asin_cat_df = asin_cat_df.withColumnRenamed('asin_id', 'dict_col').withColumnRenamed('cat_id', 'dict_col_id')

        # for create noclk_hist_category, we should create a dict to mapping from asin to its cat_id
        df = categorify_dien_data(df, uid_dict_df, mid_dict_df, cat_dict_df, proc, "local_train_splitByUser.parquet")
        df = collapse_by_hist(df, proc, "collapsed", min_num_hist = 2, max_num_hist = 100)
        df = add_negative_sample(df, categorified_dict_dfs, asin_cat_df, proc, "records_with_negative_sample")

        df = add_negative_hist_cols(df, categorified_dict_dfs, mid_dict_df, asin_cat_df,  proc, "records_with_negative_hists")
        df = save_to_local_splitByUser(df, proc, 'local_train_splitByUser')


    t1 = timer()

    print(f"Total process time is {(t1 - t0)} secs")

    ####################################
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
