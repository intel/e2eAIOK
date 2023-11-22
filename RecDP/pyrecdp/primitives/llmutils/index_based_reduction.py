import argparse
import os
from pyrecdp.core.utils import Timer
from pyrecdp.data_processor import DataProcessor as SparkDataProcessor
from pyrecdp.primitives.llmutils.utils import get_target_file_list, read_parquet_pandas_to_spark
import pyspark.sql.functions as F

def index_based_reduction_spk(src_df, dup_df, enable_hash):
    if enable_hash:
        dest_df = src_df.join(dup_df, ["doc_id", "hash"], "anti")
    else:
        dest_df = src_df.join(dup_df, "doc_id", "anti")
    return dest_df

def index_based_reduction(in_dir, dup, out_dir, enable_hash = True, spark = None):
    if spark == None:
        rdp = SparkDataProcessor()
        spark=rdp.spark
    
    list_files = get_target_file_list(in_dir, 'parquet')
    list_files = [os.path.join(in_dir, f) for f in list_files]
    large_files = [f for f in list_files if os.path.getsize(f) > 2**30]
    
    if len(large_files) > 0:
        small_files = [f for f in list_files if f not in large_files]
        first = True
        if len(small_files):
            src_df = spark.read.parquet(small_files)
            first = False
        src_df_large = read_parquet_pandas_to_spark(large_files, spark)
        if first:
            src_df = src_df_large
        else:
            src_df = src_df.union(src_df_large)
    else:    
        src_df = spark.read.option("recursiveFileLookup", "true").parquet(in_dir)
    dup_df = spark.read.option("recursiveFileLookup", "true").parquet(dup)
    
    before_total = src_df.count()
    dest_df = index_based_reduction_spk(src_df, dup_df, enable_hash)
    dest_df.write.mode('overwrite').parquet(f"{out_dir}")
    dest_df = spark.read.parquet(f"{out_dir}")
    dest_total = dest_df.count()
    print(f"Before dedup num_docs is {before_total}")
    print(f"after dedup num_docs is {dest_total}")
    print(f"file saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", dest="in_dir", type=str)
    parser.add_argument("-dup", dest="dup", type=str)
    parser.add_argument("-out", dest="out_dir", type=str)
    parser.add_argument("--hashkey", dest="enable_hash", action="store_true")
    args = parser.parse_args()
    
    in_dir = args.in_dir
    dup = args.dup
    out_dir = args.out_dir
    enable_hash = args.enable_hash
    print(f"enable hash is {enable_hash}")
    
    with Timer(f"apply duplicates.pickle to create new data"):
        index_based_reduction(in_dir, dup, out_dir, enable_hash)