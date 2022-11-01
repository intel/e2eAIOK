import init

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
        op_gen_dict = GenerateDictionary(to_categorify_cols, isParquet=True)
        proc.reset_ops([op_gen_dict])
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        # or we can simply load from pre-gened
        dict_dfs = [{'col_name': name, 'dict': proc.spark.read.format("arrow").load(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in to_categorify_cols]    

    if enable_freqlimit:
        dict_dfs = [{'col_name': dict_df['col_name'], 'dict': dict_df['dict'].filter('count >= 15')} for dict_df in dict_dfs]
    else:
        dict_dfs = [{'col_name': dict_df['col_name'], 'dict': dict_df['dict'].filter('dict_col_id > 0')} for dict_df in dict_dfs]

    # start to do categorify
    op_categorify = Categorify(to_categorify_cols, dict_dfs=dict_dfs)
    op_fillna_for_categorified = FillNA(to_categorify_cols, 0)
    proc.append_ops([op_categorify, op_fillna_for_categorified])
    t1 = timer()
    df = proc.transform(df, name=output_name, df_cnt = 4373472329)
    t2 = timer()
    print("Categorify took %.3f" % (t2 - t1))
    
    return df


def main():
    path_prefix = "hdfs://"
    current_path = "/dlrm_nativesql/"
    csv_folder = "/dlrm/csv_raw_data"
    parquet_folder = "/dlrm_nativesql/dlrm_parquet"
    #path = os.path.join(path_prefix, file)

    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
    native_sql_path = "/home/vmagent/app/native-sql-engine/native-sql-engine/core/target/spark-columnar-core-1.2.0-snapshot-jar-with-dependencies.jar"
    native_arrow_datasource_path = "/home/vmagent/app/native-sql-engine/arrow-data-source/standard/target/spark-arrow-datasource-standard-1.2.0-snapshot-jar-with-dependencies.jar"

    exec_scala_udf_jars = "/mnt/nvme2/chendi/BlueWhale/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
    exec_native_sql_path = "/mnt/nvme2/chendi/intel-bigdata/OAP/native-sql-engine/native-sql-engine/core/target/spark-columnar-core-1.2.0-snapshot-jar-with-dependencies.jar"
    exec_native_arrow_datasource_path = "/mnt/nvme2/chendi/intel-bigdata/OAP/native-sql-engine/arrow-data-source/standard/target/spark-arrow-datasource-standard-1.2.0-snapshot-jar-with-dependencies.jar"

    ##### 1. Start spark and initialize data processor #####
    t0 = timer()
    spark = SparkSession.builder.master('yarn')\
        .appName("DLRM_nativesql")\
        .config("spark.memory.offHeap.size", "600G")\
        .config("spark.driver.memory", "30G")\
        .config("spark.driver.memoryOverhead", "30G")\
        .config("spark.executor.instances", "10")\
        .config("spark.executor.cores", "6")\
        .config("spark.executor.memory", "30G")\
        .config("spark.executor.memoryOverhead", "20G")\
        .config("spark.sql.broadcastTimeout", "7200")\
        .config("spark.cleaner.periodicGC.interval", "60min")\
        .config("spark.driver.extraClassPath", 
                f"{exec_native_sql_path}:{exec_native_arrow_datasource_path}:{exec_scala_udf_jars}")\
        .config("spark.executor.extraClassPath", 
                f"{exec_native_sql_path}:{exec_native_arrow_datasource_path}:{exec_scala_udf_jars}")\
        .config("spark.sql.extensions", "com.intel.oap.ColumnarPlugin, com.intel.oap.spark.sql.ArrowWriteExtension")\
        .config("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")\
        .config("spark.sql.execution.sort.spillThreshold", "536870912")\
        .config("spark.oap.sql.columnar.sortmergejoin.lazyread", "true")\
        .config("spark.sql.adaptive.enabled", "true")\
        .getOrCreate()

    files = ["day_%d" % i for i in range(0, 24)]
    #files = ["day_0"]
    file_names = [os.path.join(path_prefix, parquet_folder, filename) for filename in files]

    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", enable_gazelle=True, spark_mode='standalone')
    #df = spark.read.format("arrow").option("originalFormat", "csv").option('sep', '\t').load(file_names)
    df = spark.read.format("arrow").load(file_names)
    df = categorifyAllFeatures(df, proc, output_name="dlrm_categorified", gen_dict=False, enable_freqlimit=False)
    t1 = timer()

    print(f"Total process time is {(t1 - t0)} secs")


if __name__ == "__main__":
    main()
