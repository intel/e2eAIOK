#!/env/bin/python

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pyrecdp
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyrecdp.data_processor import *
from pyrecdp.utils import *
from pyspark import *
from pyspark.sql import *

def main():
    path_prefix = "file://"
    cur_folder = str(pathlib.Path(__file__).parent.absolute())
    folder = cur_folder + "/data"
    path = path_prefix + folder
    recdp_path = pyrecdp.__path__[0]
    scala_udf_jars = recdp_path + "/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
    print(scala_udf_jars)

    ##### 1. Start spark and initialize data processor #####
    spark = SparkSession.builder.master("local[1]")\
        .config('spark.eventLog.enabled', False)\
        .config('spark.driver.maxResultSize', '16G')\
        .config('spark.driver.memory', '10g')\
        .config('spark.worker.memory', '10g')\
        .config('spark.executor.memory', '10g')\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .config("spark.executor.extraClassPath", f"{scala_udf_jars}")\
        .appName("test_sortArrayByFrequency")\
        .getOrCreate()
    
    proc = DataProcessor(spark)
    print(f"DataSource path is {path}")
    df = spark.read.parquet(f"{path}")
    df = df.select("language", "tweet_timestamp")
    df = df.withColumn("dt_hour", f.dayofweek(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType()))
    df = df.groupby('dt_hour').agg(f.collect_list("language").alias("language_list"))
    df = df.filter("size(language_list) > 3")
    df = df.withColumn("sorted_langugage", f.expr(f"sortStringArrayByFrequency(language_list)"))
    df.printSchema()
    df.show(24, vertical = True, truncate = 100)


if __name__ == "__main__":
    main()
