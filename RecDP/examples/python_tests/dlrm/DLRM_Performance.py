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
	path_prefix = "hdfs://"
	current_path = "/dlrm/"
	csv_folder = "/dlrm/csv_raw_data"
	#path = os.path.join(path_prefix, file)

	scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

	##### 1. Start spark and initialize data processor #####
	t0 = timer()
	spark = SparkSession.builder.master('local[80]')\
		.appName("DLRM")\
		.config("spark.driver.memory", "400G")\
		.config("spark.driver.memoryOverhead", "80G")\
		.config("spark.executor.cores", "80")\
		.config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
		.getOrCreate()

	#files = ["day_%d" % i for i in range(0, 24)]
	files = ["day_0"]
	file_names = [os.path.join(path_prefix, csv_folder, filename) for filename in files]

	proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')
	df = spark.read.schema(schema).option('sep', '\t').csv(file_names)
	#df = spark.read.parquet("/dlrm/categorified_stage1")
	df = categorifyAllFeatures(df, proc, output_name="dlrm_categorified", gen_dict=True, enable_freqlimit=True)
	t1 = timer()

	print(f"Total process time is {(t1 - t0)} secs")


if __name__ == "__main__":
	main()