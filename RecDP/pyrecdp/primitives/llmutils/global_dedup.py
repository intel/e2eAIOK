import argparse
import os, sys
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.llmutils import global_hash, global_hash_spk, index_based_reduction, index_based_reduction_spk
import pyspark.sql.functions as F
from pyrecdp.core import SparkDataProcessor

def get_hash_indexing_spk(spark_df):
    dict_df_all = spark_df
    column_names = dict_df_all.columns
    if 'hash' not in column_names or 'doc_id' not in column_names or 'source' not in column_names:
        return dict_df_all, True
    dict_df_all = dict_df_all.groupby('hash').agg(F.collect_list("doc_id").alias('doc_id_list'), F.collect_list("source").alias('source_list'), F.count("hash").alias('hash_count'))
    return dict_df_all, False

def get_hash_indexing(data_dir, out_dir, spark = None):
    if spark == None:
        rdp = SparkDataProcessor()
        spark=rdp.spark
    dict_df_all = spark.read.option("recursiveFileLookup", "true").parquet(data_dir)
    total_docs = dict_df_all.count()
    
    index_df, skip = get_hash_indexing_spk(dict_df_all)
    if skip:
        return skip
    hash_total_docs = index_df.count()
    
    out_file = os.path.join(out_dir, "hash_indexing_dict")
    index_df.write.mode('overwrite').parquet(f"{out_file}")
    
    print(f"Index has been written to {out_file}")
    print(f"  Total processed documents count is {total_docs}")
    print(f"  Total distinct hash count is {hash_total_docs}")
    return False

def get_duplication_list_spk(spark_df):
    dict_df_all = spark_df
    column_names = dict_df_all.columns
    print(column_names)
    if 'hash_count' not in column_names or 'doc_id_list' not in column_names:
        return dict_df_all, True
    
    dict_df_all = dict_df_all.filter("hash_count > 1").cache()
    
    dict_df_all = dict_df_all.withColumn("doc_id_list", F.slice(F.col("doc_id_list"), 2, F.size(F.col("doc_id_list"))))\
                             .withColumn("doc_id_list", F.explode("doc_id_list"))\
                             .select(F.col("hash"), F.col("doc_id_list").alias("doc_id"))
                             
    return dict_df_all, False

def get_duplication_list(data_dir, out_dir, spark = None):
    if spark == None:
        rdp = SparkDataProcessor()
        spark=rdp.spark
    dict_df_all = spark.read.option("recursiveFileLookup", "true").parquet(data_dir)
    
    duplicate_df, skip = get_duplication_list_spk(dict_df_all)
    if skip:
        return skip
    
    out_file = os.path.join(out_dir, "duplications_index")
    duplicate_df.write.mode("overwrite").parquet(f"{out_file}")
    
    return False

def global_dedup(with_hash, data_files, data_dir, out_dir, in_type, n_parallel, is_norm):
    # 1. if input hasn't been processed by global hash
    if not with_hash:
        out_dir_tmp = os.path.join(out_dir, 'add_normhash' if is_norm else 'add_hash')
        with Timer(f"Generate Global Hash, normailization is {is_norm}"):
            global_hash("", data_files, data_dir, in_type, n_parallel, out_dir_tmp, is_norm)
            data_dir_tmp = out_dir_tmp
    else:
        data_dir_tmp = data_dir
    
    rdp = SparkDataProcessor()
    spark=rdp.spark
    
    # 2. get global hash indexing
    with Timer(f"Generate Global indexing based on hash"):
        skip = get_hash_indexing(data_dir_tmp, out_dir, spark = spark)
    if not skip:
        data_dir_tmp = os.path.join(out_dir, "hash_indexing_dict")
    else:
        data_dir_tmp = data_dir

    # 3. generate duplication indexing
    with Timer(f"Generate global duplication list"):
        skip = get_duplication_list(data_dir_tmp, out_dir, spark = spark)
    if not skip:
        data_dir_tmp = data_dir if with_hash else out_dir_tmp
        dup_dir = os.path.join(out_dir, "duplications_index")
    else:
        return
    
    # 4. deduplicate input
    with Timer(f"reduce input file based on detected duplication"):
        index_based_reduction(data_dir_tmp, dup_dir, out_dir, enable_hash = True, spark = spark)
        
    
def global_dedup_spk(spark_df, source, is_norm):
    spark = spark_df.sparkSession
    # 1. if input hasn't been processed by global hash
    with Timer(f"Generate Global Hash, normailization is {is_norm}"):
        hash_df = global_hash_spk(spark_df, source, is_norm).cache()
        post_global_hash_count = hash_df.count()
    
    # 2. get global hash indexing
    with Timer(f"Generate Global indexing based on hash"):
        ret_df, skip = get_hash_indexing_spk(hash_df)
        if not skip:
            ret_df = ret_df.cache()
            index_count = ret_df.count()

    # 3. generate duplication indexing
    with Timer(f"Generate global duplication list"):
        ret_df, skip = get_duplication_list_spk(ret_df)
        if not skip:
            ret_df = ret_df.cache()
            duplication_count = ret_df.count()
    
    # 4. deduplicate input
    with Timer(f"reduce input file based on detected duplication"):
        out_df = index_based_reduction_spk(hash_df, ret_df, True).cache()
        post_global_dedup_count = out_df.count()
        
    print(f"Input data count is {post_global_hash_count}")
    print(f"  unique data count is {index_count}")
    print(f"  duplication count is {duplication_count}")
    print(f"  post-deduplication count is {post_global_dedup_count}")
        
    return out_df