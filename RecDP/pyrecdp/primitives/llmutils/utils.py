import os
from multiprocessing import Pool, cpu_count
from math import ceil
from tqdm import tqdm
import ftfy
import string
import re
from pyspark.sql.functions import input_file_name
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row

def convert_listoflist_to_spk(components, spark):
    # convert components to spark df
    R = Row('component')
    components_sdf = spark.createDataFrame([R(c) for c in components])
    return components_sdf

def read_json(data_files, spark, rowid = False):
    schema = StructType([ 
        StructField("text",StringType(),True), 
        StructField("meta",StringType(),True)
      ])

    first = True
    for filename in data_files:
        print(filename)
        df = spark.read.text(filename)
        basename = os.path.basename(filename)
        
        if rowid:
            df = df.withColumn("__id__", F.monotonically_increasing_id())
            df_rid = df.select('__id__').withColumn("rid", F.row_number().over(Window.orderBy(F.col("__id__"))))
            df_rid = df_rid.withColumn("filename", F.lit(basename))
            df_rid = df_rid.withColumn("filename_docid", F.concat_ws("@", "filename", "rid"))
            df = df.join(df_rid.select("__id__", "filename_docid"), "__id__", "left")
        else:
            df_rid = df.withColumn("__id__", F.monotonically_increasing_id())
            df_rid = df_rid.withColumn("filename", F.lit(basename))
            df_rid = df_rid.withColumn("filename_docid", F.concat_ws("@", "filename", "__id__"))
            df = df_rid.select('value', 'filename_docid')
        
        df = df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*", "filename_docid")  
        df = df.select("filename_docid", "text", "meta")

        if first:
            first = False
            ret_df = df
        else:
            ret_df = ret_df.union(df)
    return ret_df

def global_unique_id(df, col_name):
    ret_df = df.withColumn("__id__", F.monotonically_increasing_id())
    ret_df = ret_df.withColumn(col_name, F.concat_ws("@", F.lit("global_id"), "__id__"))
    return ret_df

def get_data_files(data_dir):
    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    files = [os.path.join(data_dir, i) for i in files]
    return files

def get_nchunks_and_nproc(n_tasks, n_part = -1):
    n_proc = cpu_count()
    if n_part != -1 and n_part < n_proc:
        n_proc = n_part
    n_chunks = ceil(n_tasks / n_proc)
    remain = n_tasks % n_proc
    if n_chunks == 1 and remain:
        n_proc = remain
    return n_chunks, n_proc

def launch_mp(n_proc, args, callable):
    print(f"parallelize with {n_proc} processes")
    
    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(callable, args), total=len(args),
        )
        for test in pbar:
            pbar.update()
            if test:
                continue
         
def normalize_str(s):
    s = ftfy.fix_text(s, normalization="NFC")
    return s

def clean_str(s):
    s = normalize_str(s)
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s.strip())
    return s
