from pyrecdp.data_processor import *
from pyrecdp.utils import *

import logging
from timeit import default_timer as timer
import os
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd

###############################################
# !!!put HDFS NODE here, empty won't proceed!!!
HDFS_NODE = "1"
###############################################

# Define Schema
LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]
schema = StructType(label_fields + int_fields + str_fields)

def categorifyAllFeatures(df, proc, output_name="categorified"):
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]    
    dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
        "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in to_categorify_cols]
    print([i['dict'].count() for i in dict_dfs])

    # start to do categorify
    op_categorify = Categorify(to_categorify_cols, dict_dfs=dict_dfs)
    op_sort = Sort(["monotonically_increasing_id"])
    proc.reset_ops([op_categorify, op_sort])
    
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Categorify took %.3f" % (t2 - t1)) 
    return df

def generate_dicts(spark, path_list, proc):
    dict_dfs = []
    first = True
    for file in path_list:
        df_single = spark.read.parquet(file)
        df_single = df_single.orderBy("monotonically_increasing_id")
        if first:
            first = False
            df = df_single
        else:
            df = df.union(df_single)
    # only call below function when target dicts were not pre-prepared        
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]
    op_gen_dict = GenerateDictionary(to_categorify_cols, id_by_count = False)
    proc.reset_ops([op_gen_dict])
    t1 = timer()
    dict_dfs = proc.generate_dicts(df)
    t2 = timer()
    print("Generate Dictionary took %.3f" % (t2 - t1))
    
    print([i['dict'].count() for i in dict_dfs])

def main(hdfs_node, dataset_path):
    import os
    host_name = os.uname()[1]
    print(host_name)
    if hdfs_node == "1":
        path_prefix = f"file://"
        total_days = 1
    else:
        path_prefix = f"hdfs://{hdfs_node}:9000"
        total_days = 23
    current_path = f"{dataset_path}/output/"

    scala_udf_jars = "/opt/intel/oneapi/intelpython/latest/lib/python3.7/site-packages/ScalaProcessUtils/built/31/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    ##### 2. Start spark and initialize data processor #####
    t1 = timer()
    spark = SparkSession.builder.master(f'spark://{host_name}:7077')\
        .appName("DLRM")\
        .config("spark.driver.memory", "20G")\
        .config("spark.driver.memoryOverhead", "10G")\
        .config("spark.executor.instances", "4")\
        .config("spark.executor.cores", "16")\
        .config("spark.executor.memory", "100G")\
        .config("spark.executor.memoryOverhead", "20G")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .config("spark.executor.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="800GB", spark_mode='standalone')

    
    train_files = ["day_%d" % i for i in range(0, total_days)]

    #############################
    # 1. Process category columns
    #############################
    max_ind_range = 40000000
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]
    int_cols = ['_c%d' % i for i in INT_COLS]    
    label_cols = ['_c0']    
    op_mod = FeatureModification(dict((i, f"(f.conv(f.col('{i}'), 16, 10) % {max_ind_range}).cast('int')") for i in to_categorify_cols), op = 'inline')
    op_fillna_for_categorified = FillNA(to_categorify_cols, 0)
    op_fillna_for_label = FillNA(label_cols, 0)
    op_fillna_for_int = FillNA(int_cols, 0)
    op_fillnegative_for_int = FeatureModification(dict((i, f"f.when(f.col('{i}') < 0, 0).otherwise(f.col('{i}'))") for i in int_cols), op = "inline")
    for filename in train_files:
        t11 = timer()
        proc.reset_ops([op_mod, op_fillna_for_categorified, op_fillna_for_label, op_fillna_for_int, op_fillnegative_for_int])
        train_df = spark.read.parquet(f"{path_prefix}{current_path}/dlrm_parquet_train_{filename}")
        train_df = proc.transform(train_df, name=f"dlrm_parquet_train_proc_{filename}")
        t12 = timer()
        print(f"Process {filename} categorified columns completed, took {(t12 - t11)} secs")

    t11 = timer()
    proc.reset_ops([op_mod, op_fillna_for_categorified, op_fillna_for_label, op_fillna_for_int, op_fillnegative_for_int])
    test_df = spark.read.parquet(f"{path_prefix}{current_path}/dlrm_parquet_test")
    test_df = proc.transform(test_df, name="dlrm_parquet_test_proc")
    t12 = timer()
    print(f"Process test categorified columns completed, took {(t12 - t11)} secs")

    t11 = timer()
    proc.reset_ops([op_mod, op_fillna_for_categorified, op_fillna_for_label, op_fillna_for_int, op_fillnegative_for_int])
    valid_df = spark.read.parquet(f"{path_prefix}{current_path}/dlrm_parquet_valid")
    valid_df = proc.transform(valid_df, name="dlrm_parquet_valid_proc")
    t12 = timer()
    print(f"Process valid categorified columns completed, took {(t12 - t11)} secs")

    #############################
    # 2. generate dict
    #############################
    path_list = [f"{path_prefix}{current_path}/dlrm_parquet_train_proc_{filename}" for filename in train_files]
    path_list += [f"{path_prefix}{current_path}/dlrm_parquet_test_proc", f"{path_prefix}{current_path}/dlrm_parquet_valid_proc"]
    generate_dicts(spark, path_list, proc)

    #############################
    # 3. Apply dicts to all days
    #############################
    for filename in train_files:
        t11 = timer()
        train_df = spark.read.parquet(f"{path_prefix}{current_path}/dlrm_parquet_train_proc_{filename}")
        categorifyAllFeatures(train_df, proc, output_name=f"dlrm_categorified_{filename}")
        t12 = timer()
        print(f"Apply dicts to {filename} completed, took {(t12 - t11)} secs")
    
    t11 = timer()
    train_df = spark.read.parquet(f"{path_prefix}{current_path}/dlrm_parquet_test_proc")
    categorifyAllFeatures(train_df, proc, output_name=f"dlrm_categorified_test")
    t12 = timer()
    print(f"Apply dicts to test completed, took {(t12 - t11)} secs")
    t11 = timer()
    train_df = spark.read.parquet(f"{path_prefix}{current_path}/dlrm_parquet_valid_proc")
    categorifyAllFeatures(train_df, proc, output_name=f"dlrm_categorified_valid")
    t12 = timer()
    print(f"Apply dicts to valid completed, took {(t12 - t11)} secs")
    
    t3 = timer()
    print(f"Total process time is {(t3 - t1)} secs")

def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path',type=str,default="/home/vmagent/app/dataset/criteo/",help='dataset path for criteo')
    parser.add_argument('--local_small', action='store_true', help='worker host list')

    return parser.parse_args(args)

if __name__ == "__main__":
    import sys
    input_args = parse_args(sys.argv[1:])
    if input_args.local_small:
        main("1", input_args.dataset_path)
    else:
        if HDFS_NODE == "":
            print("Please add correct HDFS_NODE name in this file, or this script won't be able to process")
        else:
            main(HDFS_NODE, input_args.dataset_path)