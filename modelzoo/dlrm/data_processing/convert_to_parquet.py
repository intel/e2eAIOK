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
HDFS_NODE = ""
###############################################

# Define Schema
LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]
schema = StructType(label_fields + int_fields + str_fields)


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
    csv_folder =  f"{dataset_path}/raw_data/"

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
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="800GB", spark_mode='standalone')

    #############################
    # 1. convert csv to parquet
    #############################
    train_files = ["day_%d" % i for i in range(0, total_days)]
    for filename in train_files:
        t11 = timer()
        file_name = f"file://{csv_folder}{filename}"
        train_df = spark.read.schema(schema).option('sep', '\t').csv(file_name)
        train_df = train_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
        train_df = proc.transform(train_df, name=f"dlrm_parquet_train_{filename}")
        t12 = timer()
        print(f"Convert {filename} to parquet completed, took {(t12 - t11)} secs")

    import subprocess
    process = subprocess.Popen(["sh", "raw_test_split.sh", csv_folder])
    t11 = timer()
    process.wait()
    t12 = timer()
    print(f"Split day_23 to test and valid completed, took {(t12 - t11)} secs")

    t11 = timer()
    test_files = ["test/day_23"]
    test_file_names = [f"file://{csv_folder}{filename}" for filename in test_files]
    test_df = spark.read.schema(schema).option('sep', '\t').csv(test_file_names)
    test_df = test_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    test_df = proc.transform(test_df, name="dlrm_parquet_test")
    t12 = timer()
    print(f"Convert test to parquet completed, took {(t12 - t11)} secs")

    t11 = timer()
    valid_files = ["validation/day_23"]
    valid_file_names = [f"file://{csv_folder}{filename}" for filename in valid_files]
    valid_df = spark.read.schema(schema).option('sep', '\t').csv(valid_file_names)
    valid_df = valid_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    valid_df = proc.transform(valid_df, name="dlrm_parquet_valid")
    t12 = timer()
    print(f"Convert valid to parquet completed, took {(t12 - t11)} secs")

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