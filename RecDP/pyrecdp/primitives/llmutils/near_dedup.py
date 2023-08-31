import argparse
import os
import sys

import re
import numpy as np
import pickle
from pyrecdp.core.utils import Timer
from pyrecdp.core import SparkDataProcessor
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.spark_data_processor.utils import list_dir
import pyspark.sql.functions as F
from pyspark.sql.window import Window 
import shutil
from nltk import ngrams
from .utils import normalize_str, clean_str


cur_path = os.path.dirname(__file__)

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
THRESHOLD = 200

if not os.path.exists(os.path.join(cur_path, "third_party")):
    print(f"'third_party' is not found! please use 'cp -r third_party' {cur_path}")
    exit

from .third_party import generate_connected_components, generate_duplicates_dict
from datasketch import MinHash

def generate_hash_values(content, idx, num_perm, ngram_size, hashranges, permutations):
    # 0. apply normalization to content
    content = clean_str(content)
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    
    #1. using bigcode impl to calculate minHash
    m = MinHash(num_perm=num_perm, permutations = permutations )
    m.update_batch([token.encode('utf8') for token in tokens])
    
    #2. map results to each band
    Hs = [bytes(m.hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]

def generate_edges(nodes):
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]

def get_hash_ranges(B = None, R = None):
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    return HASH_RANGES

def convert_to_slimPJ_fmt(first, second):
    return [f"{first} :: {second}"]

def minHashLSH_prepare(df, num_perm, ngram_size, B, R):
    HASH_RANGES = get_hash_ranges(B, R)
    print(f"num_bands is {B}, ranges is {R}")
    
    pipeline = (
        df.rdd
        .flatMap(
            lambda x: generate_hash_values(
                content=x[1],
                idx=x[0],
                num_perm=num_perm,
                ngram_size=ngram_size,
                hashranges=HASH_RANGES,
                permutations = None
            )
        )
        .groupBy(lambda x: (x[0], x[1]))
        .flatMap(lambda x: generate_edges([(i[2]) for i in x[1]]))
        .flatMap(lambda x: convert_to_slimPJ_fmt(x[0], x[1]))
        .distinct()
    )
    return pipeline
    
def read_json(data_files, spark):
    from pyspark.sql.functions import input_file_name
    from pyspark.sql.types import StructType,StructField, StringType
    schema = StructType([ 
        StructField("text",StringType(),True), 
        StructField("meta",StringType(),True)
      ])

    first = True
    for filename in data_files:
        print(filename)
        df = spark.read.text(filename)
        
        df = df.withColumn("__id__", F.monotonically_increasing_id())
        df_rid = df.select('__id__').withColumn("rid", F.row_number().over(Window.orderBy(F.col("__id__"))))
        df_rid = df_rid.withColumn("filename", F.lit(os.path.basename(filename)))
        df_rid = df_rid.withColumn("filename_docid", F.concat_ws("@", "filename", "rid"))
          
        df = df.join(df_rid.select("__id__", "filename_docid"), "__id__", "left")
        
        df = df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*", "filename_docid")  
        df = df.select("filename_docid", "text", "meta")

        if first:
            first = False
            ret_df = df
        else:
            ret_df = ret_df.union(df)
    return ret_df

def filter_data(data):
    return len(clean_str(data)) >= THRESHOLD

def near_dedup(data_files, dup_dir, ngram_size, num_perm, bands, ranges):
    rdp = SparkDataProcessor()
    spark=rdp.spark  
    try:
        with Timer("Load data with RowID"):
            df = read_json(data_files, spark).cache()
            total_length = df.count()
            
        pipeline = minHashLSH_prepare(df, num_perm, ngram_size, bands, ranges)
        with Timer("generate minHashLsh"):
            if os.path.exists(dup_dir):
                shutil.rmtree(dup_dir, ignore_errors=True)
            results = pipeline.saveAsTextFile(dup_dir)
            
        
        with Timer(f"generate_connected_components all"):
            dup_connected_args = argparse.Namespace()
            dup_connected_args.input_dir = dup_dir
            dup_connected_args.out_file = os.path.join(
                dup_dir, "connected_components.pickle"
            )
            generate_connected_components.generate_connected_components_mp(
                dup_connected_args
            )
            
        with Timer(f"generate_duplicates_dict all"):
            dup_docs = os.path.join(dup_dir, "duplicates.pickle")
            dup_dict_args = argparse.Namespace()
            dup_dict_args.input_file = os.path.join(
                dup_dir, "connected_components.pickle"
            )
            dup_dict_args.out_file = dup_docs
            generate_duplicates_dict.generate_duplicates(dup_dict_args)

        with open(os.path.join(dup_dir, "duplicates.pickle"), 'rb'):
            dup_dict = pickle.load()
            dup_sum = 0
            for _, v in dup_dict.items():
                dup_sum += len(list(v))

        print(f"Completed!!")
        print(f"    total processed {total_length} documents")
        print(f"    total detected {dup_sum} duplicated documents")
        print(f"    duplicate ratio is {dup_sum/total_length}")
    except Exception as e:
        spark.stop()
        print("Failed", e)