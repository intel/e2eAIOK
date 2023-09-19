import argparse
import os
import sys

import re
import numpy as np
import pickle
from pyrecdp.core.utils import Timer
from pyrecdp.core import SparkDataProcessor
from pyrecdp.core.utils import Timer
import pyspark.sql.functions as F
from pyspark.sql import Row

import shutil
from nltk import ngrams
from .utils import normalize_str, clean_str, read_json, global_unique_id, convert_listoflist_to_spk


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

def filter_data(data):
    return len(clean_str(data)) >= THRESHOLD

def near_dedup_spk(spark_df, ngram_size, num_perm, bands, ranges):
    df = spark_df
    input_count = df.count()
    spark = df.sparkSession
    df_with_id = global_unique_id(df, 'filename_docid')
    pipeline = minHashLSH_prepare(df_with_id, num_perm, ngram_size, bands, ranges)
    with Timer("generate minHashLsh"):
        results = pipeline.collect()
        
    with Timer("generate_connected_components => duplicates"):
        components = generate_connected_components.generate_connected_components_py(results)
        duplicates = [c for c_list in components for c in c_list[1:]]
        R = Row('filename_docid')
        duplicates_sdf = spark.createDataFrame([R(dup) for dup in duplicates]).cache()
        total_dup = duplicates_sdf.count()
        
    with Timer("deduplicate input data"):
        ret = df_with_id.join(duplicates_sdf, 'filename_docid', 'anti').cache()
        ret_count = ret.count()
        
    dup_sum = input_count - ret_count
    print(f"Completed!!")
    print(f"    total processed {input_count} documents")
    print(f"    total detected {total_dup} duplicated documents, exact deduplicated counts is {dup_sum}")
    print(f"    duplicate ratio is {dup_sum/input_count}")
        
    return ret


def near_dedup(data_files, dup_dir, ngram_size, num_perm, bands, ranges):
    rdp = SparkDataProcessor()
    spark=rdp.spark
    try:
        with Timer("Load data with RowID"):
            df = read_json(data_files, spark, rowid = True).cache()
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

        with open(os.path.join(dup_dir, "duplicates.pickle"), 'rb') as f:
            dup_dict = pickle.load(f)
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