import init

import os
import pandas as pd
import numpy as np
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
from timeit import default_timer as timer
import logging
from RecsysSchema import RecsysSchema
from pyrecdp.data_processor import *
from pyrecdp.utils import *
import hashlib

def categorifyAllFeatures(df, proc, output_name="categorified", gen_dicts=False):
    # 1. define operations
    # 1.1 fill na and features
    op_fillna_str = FillNA(
        ['present_domains', 'present_links', 'hashtags'], "")

    # 1.3 categorify
    # since language dict is small, we may use udf to make partition more even
    #'present_domains', 'present_links', 
    op_categorify_multi = Categorify(
        ['present_domains', 'present_links', 'hashtags'], gen_dicts=gen_dicts, doSplit=True, keepMostFrequent=True)
    op_fillna_for_categorified = FillNA(['present_domains', 'present_links', 'hashtags'], -1)

    # transform
    proc.append_ops([op_fillna_str, op_categorify_multi, op_fillna_for_categorified])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Data Process 1 and udf categorify took %.3f" % (t2 - t1))

    return df

def main():
	path_prefix = "hdfs://"
	current_path = "/recsys2020_0608_categorify_example_1/"
	original_folder = "/recsys2021_0608/"
	dicts_folder = "recsys_dicts/"
	recsysSchema = RecsysSchema()

	scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

	##### 1. Start spark and initialize data processor #####
	t0 = timer()
	spark = SparkSession.builder.master('local[80]')\
		.appName("Recsys2020_data_process_support_for_friesian")\
		.config("spark.driver.memory", "480G")\
		.config("spark.executor.cores", "80")\
		.config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
		.getOrCreate()

	schema = recsysSchema.toStructType()

	# 1.1 prepare dataFrames
	# 1.2 create RecDP DataProcessor
	proc = DataProcessor(spark, path_prefix,
			current_path=current_path, dicts_path=dicts_folder, shuffle_disk_capacity="1200GB", spark_mode='local')

	# ===============================================
	# basic: Do categorify for all columns
	df = spark.read.parquet(path_prefix + original_folder)

	# rename firstly
	df = df.withColumnRenamed('enaging_user_following_count', 'engaging_user_following_count')
	df = df.withColumnRenamed('enaging_user_is_verified', 'engaging_user_is_verified')
	df = categorifyAllFeatures(df, proc, gen_dicts=True)

	t1 = timer()

	print(f"Total process time is {(t1 - t0)/1000000} secs")


if __name__ == "__main__":
    main()