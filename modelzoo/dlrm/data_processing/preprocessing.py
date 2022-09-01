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

# Define Schema
LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]
schema = StructType(label_fields + int_fields + str_fields)

to_be_categorified = [23, 35, 14, 33]

def categorifyAllFeatures(df, proc, output_name="categorified", gen_dict=False, enable_freqlimit=False):
    dict_dfs = []
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]
    #to_categorify_cols = ['_c%d' % i for i in to_be_categorified]
    if gen_dict:
        # only call below function when target dicts were not pre-prepared        
        op_gen_dict = GenerateDictionary(to_categorify_cols, isParquet=False)
        proc.reset_ops([op_gen_dict])
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        # or we can simply load from pre-gened
        dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in to_categorify_cols]

    print([i['dict'].count() for i in dict_dfs])

    if enable_freqlimit:
        dict_dfs = [{'col_name': dict_df['col_name'], 'dict': dict_df['dict'].filter('count >= 15')} for dict_df in dict_dfs]

    # start to do categorify
    op_categorify = Categorify(to_categorify_cols, dict_dfs=dict_dfs)
    op_fillna_for_categorified = FillNA(to_categorify_cols, 0)
    proc.append_ops([op_categorify, op_fillna_for_categorified])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Categorify took %.3f" % (t2 - t1))
    
    return df

def main():
    import os
    host_name = os.uname()[1]
    print(host_name)
    path_prefix = "file://"
    current_path = "/home/vmagent/app/dataset/criteo/output/"
    csv_folder = "/home/vmagent/app/raw_data/"

    scala_udf_jars = "/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/ScalaProcessUtils/built/31/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    ##### 2. Start spark and initialize data processor #####
    t1 = timer()
    spark = SparkSession.builder.master(f'spark://{host_name}:7077')\
        .appName("DLRM")\
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
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="550GB", spark_mode='standalone')

    train_files = ["day_%d" % i for i in range(0, 23)]
    file_names = [f"{path_prefix}{csv_folder}{filename}" for filename in train_files]
    df = spark.read.schema(schema).option('sep', '\t').csv(file_names)
    df = categorifyAllFeatures(df, proc, output_name="dlrm_categorified", gen_dict=True, enable_freqlimit=False)
    #df = categorifyAllFeatures(df, proc, output_name="dlrm_categorified", gen_dict=False, enable_freqlimit=False)
    t2 = timer()
    print(f"Train data process time is {(t2 - t1)} secs")

    # for valid + test data
    import subprocess
    process = subprocess.Popen(["sh", "raw_test_split.sh", csv_folder])
    process.wait()
    test_files = ["test/day_23"]
    test_file_names = [f"{path_prefix}{csv_folder}{filename}" for filename in test_files]
    test_df = spark.read.schema(schema).option('sep', '\t').csv(test_file_names)
    test_df = categorifyAllFeatures(test_df, proc, output_name="dlrm_categorified_test", gen_dict=False, enable_freqlimit=False)

    valid_files = ["validation/day_23"]
    valid_file_names = [f"{path_prefix}{csv_folder}{filename}" for filename in valid_files]
    valid_df = spark.read.schema(schema).option('sep', '\t').csv(valid_file_names)
    valid_df = categorifyAllFeatures(valid_df, proc, output_name="dlrm_categorified_valid", gen_dict=False, enable_freqlimit=False)
    t3 = timer()

    print(f"Total process time is {(t3 - t1)} secs")


if __name__ == "__main__":
    main()
    