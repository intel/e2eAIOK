import findspark
findspark.init()

import os
import sys

import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

import timeit


spark = SparkSession.builder.master('yarn')\
        .appName("Recsys2021_data_process")\
        .getOrCreate()

start = timeit.timeit()
ks.set_option('compute.max_rows', 10000000)
kdf = ks.read_parquet('/recsys2021_0608')
kdf = ks.read_parquet("/recsys2021_0608_processed/first_hashtags")
#kser = kdf['language'].fillna("").factorize()[0]
kser = kdf['a'].fillna("").factorize()[0]

ks.DataFrame(kser).to_parquet('/koalas_example/hashtags')

end = timeit.timeit()
print("koalas categorify took %.6f secs" % (end - start) * 1000000)
