import os, sys
from multiprocessing import Pool, cpu_count
from math import ceil
import subprocess
from tqdm import tqdm
import pandas as pd
import string
import re
import urllib.error
import ftfy

from pyspark.sql.functions import input_file_name
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row
import subprocess
import time

def convert_listoflist_to_spk(components, spark):
    # convert components to spark df
    R = Row('component')
    components_sdf = spark.createDataFrame([R(c) for c in components])
    return components_sdf

def read_parquet(data_files, spark, rowid = False):
    first = True
    dirname_list = [os.path.dirname(f) for f in data_files]
    common = os.path.commonprefix(dirname_list)
    for filename in data_files:
        df = spark.read.parquet(filename)
        basename = os.path.basename(filename)
        prefix = os.path.dirname(filename)
        basename = os.path.join(prefix.replace(common, ''), basename)
        
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
            df = df_rid.drop('__id__', 'filename')
        
        df = df.select("filename_docid", "text", "meta")

        if first:
            first = False
            ret_df = df
        else:
            ret_df = ret_df.union(df)
    return ret_df


def read_json(data_files, spark, rowid = False):
    schema = StructType([ 
        StructField("text",StringType(),True), 
        StructField("meta",StringType(),True)
      ])

    first = True
    dirname_list = [os.path.dirname(f) for f in data_files]
    common = os.path.commonprefix(dirname_list)
    for filename in data_files:
        df = spark.read.text(filename)
        basename = os.path.basename(filename)
        prefix = os.path.dirname(filename)
        basename = os.path.join(prefix.replace(common, ''), basename).replace("/", "_")
        
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
    ret_df = df
    if 'filename_docid' in df.schema.names:
        ret_df = ret_df.withColumn(col_name, F.regexp_replace(F.col("filename_docid"), "/", "_"))
        return ret_df
    if col_name in df.schema.names:
        return ret_df
    ret_df = ret_df.select(F.concat_ws("@", F.lit("global_id"), F.monotonically_increasing_id()).alias(col_name), "*")
    return ret_df


def get_data_files(data_dir):
    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    files = [os.path.join(data_dir, i) for i in files]
    return files

def sub_task_per_folder(file_list):
    sub_task = {}
    for f in file_list:
        dir_name = os.path.dirname(f)
        if dir_name not in sub_task:
            sub_task[dir_name] = []
        sub_task[dir_name].append(f)
    return sub_task

def get_target_file_list_from_local(data_dir, file_type):
    if not os.path.isdir(data_dir):
        data_dir = os.path.dirname(data_dir)
        cmd = ["find", data_dir, "-name", f"{file_type}"]
    else:
        cmd = ["find", data_dir, "-name", f"*.{file_type}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    exitcode = proc.returncode
    if exitcode != 0:
        return []
    else:
        ret = stdout.decode("utf-8").split('\n')[:-1]
        ret = [i.replace(data_dir, "") for i in ret]
        ret = [i[1:] if i[0] == '/' else i for i in ret]
        return ret
    
def get_target_file_list_from_hdfs(data_dir, file_type):
    cmd = ["hdfs", "dfs",  "-find", data_dir, "-name", f"*.{file_type}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    exitcode = proc.returncode
    if exitcode != 0:
        return []
    else:
        ret = stdout.decode("utf-8").split('\n')[:-1]
        ret = [i.replace(data_dir, "") for i in ret]
        ret = [i[1:] if i[0] == '/' else i for i in ret]
        return ret
    
def get_target_file_list(data_dir, file_type, file_system_prefix = ""):
    if file_system_prefix == "file://":
        return get_target_file_list_from_local(data_dir, file_type)
    if file_system_prefix == "hdfs://":
        return get_target_file_list_from_hdfs(data_dir, file_type)
    file_list = []
    try:
        file_list = get_target_file_list_from_local(data_dir, file_type)
    except:
        file_list = []
    if len(file_list) == 0:
        try:
            file_list = get_target_file_list_from_hdfs(data_dir, file_type)
        except:
            file_list = []
    return file_list    

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
    try:
        s = normalize_str(s)
    except:
        s = ""
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s.strip())
    return s
  
def get_llmutils_home():
    return os.path.abspath(os.path.dirname(__file__))

 
def download_file(remote_path, target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            wget.download(remote_path, out=target_path)
        except urllib.error.HTTPError as e:
            print("Failed to download the file. Please check the url and network.")
            raise e

def read_parquet_pandas_to_spark(f_name_list, spark):
    first = True
    for f_name in f_name_list:
        pdf = pd.read_parquet(f_name)
        df = spark.createDataFrame(pdf)
        if first:
            first = False
            src_df = df
        else:
            src_df = src_df.union(df)
    return src_df
                
class MultiProcessManager:
    def wait_and_check(self, pool, base_script_name):
        for proc_id, (process, cmd) in pool.items():
            std_out, std_err = process.communicate()
            rc = process.wait()
            if rc != 0:
                file_name = f"{base_script_name}-proc-{proc_id}.error.log"
                print(f"Task failed, please check {file_name} for detail information")
                with open(file_name, "a") as f:
                    f.write(f"=== {time.ctime()} {' '.join(cmd)} failed. ===\n")
                    f.write(std_err.decode(sys.getfilesystemencoding()))
                    f.write("\n")
                
                
    def launch_cmdline_mp(self, args, mp, script_name):
        pool = {}
        inflight = 0
        base_script_name = os.path.basename(script_name)
        for proc_id, arg in tqdm(enumerate(args), total=len(args), desc=base_script_name):
            proc_id, x_list = arg
            cmd = ["python", script_name]
            cmd += x_list
            inflight += 1

            pool[proc_id] = (subprocess.Popen(cmd , stdout=subprocess.PIPE, stderr=subprocess.PIPE), cmd)
            
            if inflight >= mp:
                self.wait_and_check(pool, base_script_name)
                inflight = 0
                pool = {}
        self.wait_and_check(pool, base_script_name)


def read_data(data_dir, data_files, data_file_type, spark, file_system_prefix=""):
    if data_file_type in ["json", "jsonl"]:
        read_function = spark.read.json
    elif data_file_type == "parquet":
        read_function = spark.read.parquet
    else:
        read_function = spark.read.text

    file_dict = {}
    for file in data_files:
        if file_dict.get(os.path.dirname(file)) is None:
            file_dict[os.path.dirname(file)] = [file]
        else:
            file_dict[os.path.dirname(file)].append(file)
    df_dict = {}
    for parent_dir, files in file_dict.items():
        first = True
        for file in files:
            df = read_function(f"{file_system_prefix}{os.path.join(data_dir, file)}")
            if first:
                first = False
                ret_df = df
            else:
                ret_df = ret_df.union(df)
        df_dict[parent_dir] = ret_df

    return df_dict
