#!/env/bin/python

import pathlib

import pyrecdp
import pyspark.sql.functions as f
from pyrecdp.data_processor import *
from pyrecdp.utils import *
from pyspark import *
from pyspark.sql import *


def main():
    path_prefix = "file://"
    cur_folder = str(pathlib.Path(__file__).parent.absolute())
    folder = cur_folder + "/../test_data"
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
        .appName("test_categorify")\
        .getOrCreate()
    
    proc = DataProcessor(spark, path_prefix, cur_folder)
    print(f"DataSource path is {path}")
    df = spark.read.parquet(f"{path}")
    df = df.select("tweet", "language")
    df.printSchema()

    proc.reset_ops([Categorify(['language'])])
    df = proc.apply(df)
    df = df.fillna('None')
    df.show(vertical = True)
    df.explain()


if __name__ == "__main__":
    main()
