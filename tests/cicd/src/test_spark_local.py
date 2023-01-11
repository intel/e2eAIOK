#!/env/bin/python

import sys
import pathlib
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hostname',
        type=str,
        required=True,
        help='current host name')
    return parser.parse_args(args)

def main(hostname):
    MASTER_NODE = hostname
    path_prefix = "file://"
    cur_folder = str(pathlib.Path(__file__).parent.absolute())
    folder = cur_folder + "/../test_data"
    path = path_prefix + folder

    ##### 1. Start spark and initialize data processor #####
    spark = SparkSession.builder.master(f'spark://{MASTER_NODE}:7077')\
        .config('spark.driver.maxResultSize', '16G')\
        .config('spark.driver.memory', '10g')\
        .config('spark.executor.memory', '20g')\
        .config('spark.executor.instances', '20')\
        .config('spark.executor.cores', '2')\
        .appName("Recsys2021_DATA_PROCESS")\
        .getOrCreate()
    
    print(f"DataSource path is {path}")
    df = spark.read.parquet(f"{path}")
    df.printSchema()
    df = df.groupby('tweet_id').count().show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.hostname)