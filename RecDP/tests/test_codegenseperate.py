#!/env/bin/python

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pyrecdp
import pyspark.sql.functions as f
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
        .appName("test_codegenseperator")\
        .getOrCreate()
    
    proc = DataProcessor(spark)
    print(f"DataSource path is {path}")
    df = spark.read.parquet(f"{path}")
    df.printSchema()

    dict_df = df.filter("like_timestamp is not null").groupBy("engaged_with_user_id").agg({"like_timestamp": "count"})

    df.join(dict_df, "engaged_with_user_id")
    vspark = str(spark.version)
    if vspark.startswith("3.1") or vspark.startswith("3.0"):
        df = df.withColumn("engaged_with_user_id", f.expr(f"CodegenSeparator1(engaged_with_user_id)"))
    df = df.fillna('None')
    df.show(1, vertical = True)
    df.explain()


if __name__ == "__main__":
    main()
