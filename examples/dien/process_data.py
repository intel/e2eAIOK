from python_script.init_spark import *
from python_script.utils import *
from python_script.data_processor import DataProcessor
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession

dir_path = "/mnt/nvme2/chendi/BlueWhale/ai-matrix/macro_benchmark/DIEN_INTEL_TF2/pyspark_data/"
local_prefix_src = "file://" + dir_path

t0 = timer()
spark = SparkSession\
    .builder\
    .master('yarn')\
    .appName("DIEN_DATA_PREPARE") \
    .getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
t1 = timer()
# Load reviews_info and item_info from HDFS
reviews_info_df, item_info_df = load_csv(spark, local_prefix_src)
data_processor = DataProcessor(
    spark, reviews_info_df, item_info_df, dir_path)
data_processor.process()
t3 = timer()
print("\n==================== Process Time =======================\n")
print("Total process took %.3f secs" % (t3 - t0))
print("Details:")
print("start spark took %.3f secs" % (t1 - t0))
print("process and save took %.3f secs, includes:" % (t3 - t1))
for key, value in data_processor.elapse_time.items():
    print("\t%s %.3f" % (key, value))
print("\n==========================================================")
