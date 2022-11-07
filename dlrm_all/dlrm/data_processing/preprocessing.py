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
import raydp
import argparse
import yaml
from json import dumps


# Define Schema
LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]
schema = StructType(label_fields + int_fields + str_fields)

def categorifyAllFeatures(df, proc, output_name="categorified"):
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]    
    dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
        "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in to_categorify_cols]
    print([i['dict'].count() for i in dict_dfs])

    # start to do categorify
    op_categorify = Categorify(to_categorify_cols, dict_dfs=dict_dfs)
    op_sort = Sort(["monotonically_increasing_id"])
    proc.reset_ops([op_categorify, op_sort])
    
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Categorify took %.3f" % (t2 - t1)) 
    return df

def generate_dicts(spark, path_list, proc):
    dict_dfs = []
    first = True
    for file in path_list:
        df_single = spark.read.parquet(file)
        df_single = df_single.orderBy("monotonically_increasing_id")
        if first:
            first = False
            df = df_single
        else:
            df = df.union(df_single)
    # only call below function when target dicts were not pre-prepared        
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]
    op_gen_dict = GenerateDictionary(to_categorify_cols, id_by_count = False)
    proc.reset_ops([op_gen_dict])
    t1 = timer()
    dict_dfs = proc.generate_dicts(df)
    t2 = timer()
    print("Generate Dictionary took %.3f" % (t2 - t1))
    print([i['dict'].count() for i in dict_dfs])
    model_size = [i['dict'].count() for i in dict_dfs]
    return model_size

def save_info(output_path, save_path, model_size):
    CAT_COLS = list(range(14, 40))
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]
    model_size = dict(zip(to_categorify_cols, model_size))
    data_info = {"train_data": os.path.join(output_path, 'train'),
                "test_data": os.path.join(output_path, 'dlrm_categorified_test'),
                "val_data": os.path.join(output_path, 'dlrm_categorified_valid'),
                "model_size": model_size}
    with open(save_path, "w") as f:
        f.write(dumps(data_info))

def main(data_config, save_path):
    import os
    host_name = os.uname()[1]
    print(host_name)
    hdfs_node = data_config["hdfs_node"]
    path_prefix = f"hdfs://{hdfs_node}:9000"
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

    
    start, end = train_days.split("-")
    train_range = list(range(int(start), int(end) + 1))
    train_files = ["day_"+str(i) for i in train_range]

    #############################
    # 1. Process category columns
    #############################
    max_ind_range = 40000000
    to_categorify_cols = ['_c%d' % i for i in CAT_COLS]
    int_cols = ['_c%d' % i for i in INT_COLS]    
    label_cols = ['_c0']    
    op_mod = FeatureModification(dict((i, f"(f.conv(f.col('{i}'), 16, 10) % {max_ind_range}).cast('int')") for i in to_categorify_cols), op = 'inline')
    op_fillna_for_categorified = FillNA(to_categorify_cols, 0)
    op_fillna_for_label = FillNA(label_cols, 0)
    op_fillna_for_int = FillNA(int_cols, 0)
    op_fillnegative_for_int = FeatureModification(dict((i, f"f.when(f.col('{i}') < 0, 0).otherwise(f.col('{i}'))") for i in int_cols), op = "inline")
    for filename in train_files:
        t11 = timer()
        proc.reset_ops([op_mod, op_fillna_for_categorified, op_fillna_for_label, op_fillna_for_int, op_fillnegative_for_int])
        train_df = spark.read.parquet(f"{path_prefix}{output_folder}/dlrm_parquet_train_{filename}")
        train_df = proc.transform(train_df, name=f"dlrm_parquet_train_proc_{filename}")
        t12 = timer()
        print(f"Process {filename} categorified columns completed, took {(t12 - t11)} secs")

    t11 = timer()
    proc.reset_ops([op_mod, op_fillna_for_categorified, op_fillna_for_label, op_fillna_for_int, op_fillnegative_for_int])
    test_df = spark.read.parquet(f"{path_prefix}{output_folder}/dlrm_parquet_test")
    test_df = proc.transform(test_df, name="dlrm_parquet_test_proc")
    t12 = timer()
    print(f"Process test categorified columns completed, took {(t12 - t11)} secs")

    t11 = timer()
    proc.reset_ops([op_mod, op_fillna_for_categorified, op_fillna_for_label, op_fillna_for_int, op_fillnegative_for_int])
    valid_df = spark.read.parquet(f"{path_prefix}{output_folder}/dlrm_parquet_valid")
    valid_df = proc.transform(valid_df, name="dlrm_parquet_valid_proc")
    t12 = timer()
    print(f"Process valid categorified columns completed, took {(t12 - t11)} secs")

    #############################
    # 2. generate dict
    #############################
    path_list = [f"{path_prefix}{output_folder}/dlrm_parquet_train_proc_{filename}" for filename in train_files]
    path_list += [f"{path_prefix}{output_folder}/dlrm_parquet_test_proc", f"{path_prefix}{output_folder}/dlrm_parquet_valid_proc"]
    model_size = generate_dicts(spark, path_list, proc)

    #############################
    # 3. save data info to file
    #############################
    save_info(f"{path_prefix}{output_folder}", save_path, model_size)

    #############################
    # 4. Apply dicts to all days
    #############################
    for filename in train_files:
        t11 = timer()
        train_df = spark.read.parquet(f"{path_prefix}{output_folder}/dlrm_parquet_train_proc_{filename}")
        categorifyAllFeatures(train_df, proc, output_name=f"dlrm_categorified_{filename}")
        t12 = timer()
        print(f"Apply dicts to {filename} completed, took {(t12 - t11)} secs")
    
    t11 = timer()
    train_df = spark.read.parquet(f"{path_prefix}{output_folder}/dlrm_parquet_test_proc")
    categorifyAllFeatures(train_df, proc, output_name=f"dlrm_categorified_test")
    t12 = timer()
    print(f"Apply dicts to test completed, took {(t12 - t11)} secs")
    t11 = timer()
    train_df = spark.read.parquet(f"{path_prefix}{output_folder}/dlrm_parquet_valid_proc")
    categorifyAllFeatures(train_df, proc, output_name=f"dlrm_categorified_valid")
    t12 = timer()
    print(f"Apply dicts to valid completed, took {(t12 - t11)} secs")
    
    t3 = timer()
    print(f"Total process time is {(t3 - t1)} secs")

    raydp.stop_spark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        data_config = config["data_preprocess"]
        main(data_config, args.save_path)
    
