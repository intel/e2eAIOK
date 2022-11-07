from pyrecdp.data_processor import *
from pyrecdp.utils import *

import logging
from timeit import default_timer as timer
import os
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import raydp
import yaml
import argparse

# Define Schema
LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]
schema = StructType(label_fields + int_fields + str_fields)

def main(data_config):
    import os
    host_name = os.uname()[1]
    print(host_name)
    hdfs_node = data_config["hdfs_node"]
    path_prefix = f"hdfs://{hdfs_node}:9000"
    train_input_folder = data_config["train_input_folder"]
    test_input_folder = data_config["test_input_folder"]
    output_folder = data_config["output_folder"]
    spark_config = data_config["spark_config"]
    train_days = data_config["train_days"]

    ##### 2. Start spark and initialize data processor #####
    t1 = timer()
    import ray
    ray.init(address="auto")
    spark = raydp.init_spark(
            app_name=spark_config["app_name"],
            num_executors=spark_config["num_executors"],
            executor_cores=spark_config["executor_cores"],
            executor_memory=spark_config["executor_memory"],
            placement_group_strategy="SPREAD",
            configs=spark_config["configs"])
    spark.sparkContext.setLogLevel("ERROR")
    proc = DataProcessor(spark, path_prefix, current_path=output_folder, shuffle_disk_capacity="800GB", spark_mode='standalone')

    #############################
    # 1. convert csv to parquet
    #############################
    start, end = train_days.split("-")
    train_range = list(range(int(start), int(end) + 1))
    train_files = ["day_"+str(i) for i in train_range]
    for filename in train_files:
        t11 = timer()
        file_name = f"{train_input_folder}{filename}"
        train_df = spark.read.schema(schema).option('sep', '\t').csv(file_name)
        train_df = train_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
        train_df = proc.transform(train_df, name=f"dlrm_parquet_train_{filename}")
        t12 = timer()
        print(f"Convert {filename} to parquet completed, took {(t12 - t11)} secs")
    

    import subprocess
    process = subprocess.Popen(["bash", "../data_processing/raw_test_split.sh", test_input_folder])
    t11 = timer()
    process.wait()
    t12 = timer()
    print(f"Split day_23 to test and valid completed, took {(t12 - t11)} secs")

    t11 = timer()
    test_files = ["test/day_23"]
    test_file_names = [f"{test_input_folder}{filename}" for filename in test_files]
    test_df = spark.read.schema(schema).option('sep', '\t').csv(test_file_names)
    test_df = test_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    test_df = proc.transform(test_df, name="dlrm_parquet_test")
    t12 = timer()
    print(f"Convert test to parquet completed, took {(t12 - t11)} secs")

    t11 = timer()
    valid_files = ["validation/day_23"]
    valid_file_names = [f"{test_input_folder}{filename}" for filename in valid_files]
    valid_df = spark.read.schema(schema).option('sep', '\t').csv(valid_file_names)
    valid_df = valid_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    valid_df = proc.transform(valid_df, name="dlrm_parquet_valid")
    t12 = timer()
    print(f"Convert valid to parquet completed, took {(t12 - t11)} secs")

    t3 = timer()
    print(f"Total process time is {(t3 - t1)} secs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--config_path', type=str, default = None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        data_config = config["data_preprocess"]
        main(data_config)
    
