"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
import os, sys
from pyrecdp.core.utils import Timer
import json
from pyrecdp.primitives.llmutils.utils import clean_str, MultiProcessManager, get_target_file_list, get_nchunks_and_nproc, global_unique_id, sub_task_per_folder, read_json, read_parquet
import hashlib
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyrecdp.data_processor import DataProcessor as SparkDataProcessor
         

def text_normalization_spk(spark_df):
    clean_str_udf = F.udf(clean_str, T.StringType())
    columns = spark_df.columns
    ret_df = spark_df
    key = 'text' if 'text' in columns else 'content'
    ret_df = ret_df.withColumn('norm_text', clean_str_udf(F.col(key)))
    return ret_df

def text_normalization(data_dir, in_type, out_dir):
    sub_task_dir = {}
    data_files = get_target_file_list(data_dir, in_type)
    sub_task_dict = sub_task_per_folder(data_files)
    
    rdp = SparkDataProcessor()
    spark=rdp.spark
    post_global_hash_count = 0
    for sub_task, data_files in sub_task_dict.items():
        with Timer(f"processing {sub_task}"):
            data_files = [os.path.join(data_dir, f) for f in data_files]
            if in_type == 'parquet':
                sub_task_dir[sub_task] = read_parquet(data_files, spark)
            elif in_type == 'jsonl':
                sub_task_dir[sub_task] = read_json(data_files, spark)
            sub_task_dir[sub_task] = text_normalization_spk(sub_task_dir[sub_task]).cache()
            post_global_hash_count += sub_task_dir[sub_task].count()
            
            out_file = os.path.join(out_dir, sub_task)
            sub_task_dir[sub_task].write.mode("overwrite").parquet(f"{out_file}")
    print(f"data is written to {out_dir}")
    print(f"  document count is {post_global_hash_count}")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("--in_type", dest="in_type", type=str, default="jsonl")
    args = parser.parse_args()
    
    # main controller
    data_dir = args.data_dir
    out_dir = args.out_dir
    in_type = args.in_type
    with Timer(f"Text Normalization to {data_dir}"):
        text_normalization(data_dir, in_type, out_dir)