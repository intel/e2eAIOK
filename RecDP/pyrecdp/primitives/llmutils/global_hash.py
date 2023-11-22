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

def sha256str(s):
    h = hashlib.sha256()
    try:
        h.update(s.encode("utf-8"))
    except UnicodeEncodeError:
        # to avoid things like \ud809\udc50\ud808\udefc\ud808\udedb
        h.update(s.encode("utf-8", "replace"))
    return h.hexdigest()
         
def process_parquet_to_parquet(in_file_name, out_file_name, base_file_name, source, is_norm):
    pdf = pd.read_parquet(in_file_name).reset_index(drop=True)
    key = 'text' if 'text' in pdf.columns else 'content'
    global_id_list = []
    hash_value_list = []
    source_list = []
    size_list = []
    for idx, text in pdf[key].items():
        if is_norm:
            norm_text = clean_str(text)
            hash_value_list.append(sha256str(norm_text))
        else:
            hash_value_list.append(sha256str(text))
        global_id_list.append(f"{base_file_name}@{idx+1}")
        source_list.append(source)
        size_list.append(len(text.encode('utf-8')))
    
    o_pdf = pd.DataFrame()
    o_pdf['text'] = pdf[key]
    if 'meta' not in pdf.columns:
        meta_cols = [i for i in pdf.columns if i != key]
        o_pdf['meta'] = pdf[meta_cols].to_dict('records')
    else:
        o_pdf['meta'] = pdf['meta']
    if 'source' not in pdf.columns:
        o_pdf['source'] = pd.Series(source_list)
    else:
        o_pdf['source'] = pdf['source']
    o_pdf['doc_id'] = pd.Series(global_id_list)
    o_pdf['hash'] = pd.Series(hash_value_list)
    o_pdf['bytesize'] = pd.Series(size_list)
    o_pdf.to_parquet(out_file_name)
    del pdf
    
def process_jsonl_to_parquet(in_file_name, out_file_name, base_file_name, source, is_norm):
    text_list = []
    meta_list = []
    global_id_list = []
    hash_value_list = []
    source_list = []
    size_list = []
    pdf = pd.DataFrame()
    with open(in_file_name, 'r') as rdr:         
        for idx, line in enumerate(rdr):
            ob = json.loads(line)
            text = ob['text']
            if is_norm:
                normtext = clean_str(text)
                hash_value_list.append(sha256str(normtext))
            else:
                hash_value_list.append(sha256str(text))
            if 'meta' not in ob:
                ob['meta'] = str({})
            text_list.append(text)                
            meta_list.append(ob['meta'])
            global_id_list.append(f"{base_file_name}@{idx+1}")
            
            if 'source' not in ob:
                source_list.append(source)
            else:
                source_list.append(ob['source'])
            size_list.append(len(text.encode('utf-8')))
    pdf['text'] = pd.Series(text_list)
    pdf['meta'] = pd.Series(meta_list)
    pdf['source'] = pd.Series(source_list)
    pdf['doc_id'] = pd.Series(global_id_list)
    pdf['hash'] = pd.Series(hash_value_list)
    pdf['bytesize'] = pd.Series(size_list)
    pdf.to_parquet(out_file_name)


def generate_hash_index(proc_id, in_type, x_list, source, is_norm):
    for x in x_list:
        try:
            in_file_name, out_file_name, base_file_name = x
            source_name = f"{source}_{base_file_name.replace('/', '_')}"
            base_file_name = os.path.basename(base_file_name)
            out_dir = os.path.dirname(out_file_name)
            os.makedirs(out_dir, exist_ok=True)
            if in_type == 'parquet':
                process_parquet_to_parquet(in_file_name, out_file_name, base_file_name, source_name, is_norm)
            elif in_type == 'jsonl':
                process_jsonl_to_parquet(in_file_name, out_file_name, base_file_name, source_name, is_norm)
            else:
                raise NotImplementedError(f"Unsupported file type {in_type}")
        except Exception as e:
            with open(f"{out_file_name}.error.log", 'w') as f:
                f.write(f"Failed to process {base_file_name}, error is {e}")
    return True

def global_hash_spk(spark_df, source, is_norm):
    clean_str_udf = F.udf(clean_str, T.StringType())
    sha256str_udf = F.udf(sha256str, T.StringType())
    bytesize_udf = F.udf(lambda x: len(x.encode('utf-8')), T.IntegerType())
    columns = spark_df.columns
    ret_df = spark_df
    ret_df = ret_df.withColumn("source", F.lit(source))
    ret_df = global_unique_id(ret_df, 'doc_id')
    key = 'text' if 'text' in columns else 'content'
    if is_norm:
        ret_df = ret_df.withColumn('hash', sha256str_udf(clean_str_udf(F.col(key))))
    else:
        ret_df = ret_df.withColumn('hash', sha256str_udf(F.col(key)))
    ret_df = ret_df.withColumn("bytesize", bytesize_udf(F.col(key)))
    return ret_df

def global_hash(source, data_dir, in_type, out_dir, is_norm):
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
            sub_task_dir[sub_task] = global_hash_spk(sub_task_dir[sub_task], source, is_norm).cache()
            post_global_hash_count += sub_task_dir[sub_task].count()
            
            out_file = os.path.join(out_dir, sub_task)
            sub_task_dir[sub_task].write.mode("overwrite").parquet(f"{out_file}")
    print(f"data is written to {out_dir}")
    print(f"  document count is {post_global_hash_count}")


def global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm):
    files = get_target_file_list(data_dir, in_type)
    if n_parallel != -1:
        n_proc = n_parallel
    else:
        _, n_proc = get_nchunks_and_nproc(len(files), n_part = n_parallel)
    print(f"resetting to {n_proc} for number of processes")
    
    args = [(idx, [i]) for idx, i in enumerate(files)]
    cmdline_args = []
    for (idx, arg) in args:
        cmd = ["--source", source, "--proc_id", f"{idx}", "--in_dir", f"{data_dir}", "--out_dir", f"{out_dir}", "--in_type", f"{in_type}", "--file_list", f"{arg}"]
        if is_norm:
            cmd.append("--normalize")
        cmdline_args.append((idx, cmd))

    mp_mgr = MultiProcessManager()
    mp_mgr.launch_cmdline_mp(cmdline_args, n_proc, f"{__file__}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="source", type=str, default = "")
    parser.add_argument("--normalize", dest="is_norm", action="store_true")
    parser.add_argument("--proc_id", dest="proc_id", type=int, default = -1)
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("--in_type", dest="in_type", type=str, default="jsonl")
    parser.add_argument("--in_dir", dest="in_dir", type=str)
    parser.add_argument("--out_dir", dest="out_dir", type=str)
    parser.add_argument("--file_list", dest="file_list", type=str)
    parser.add_argument("-mp", dest="mp", type=int, default=-1)
    args = parser.parse_args()
    
    # main controller
    if args.proc_id == -1:
        source = args.source
        data_dir = args.data_dir
        out_dir = args.out_dir
        in_type = args.in_type
        n_parallel = args.mp
        is_norm = args.is_norm
        print(f"is_norm is {is_norm}")
        files = get_target_file_list(data_dir, in_type)
        if len(files) == 0:
            print("Detect 0 files, exit here")
            sys.exit(0)
            
        with Timer(f"generate hash to {data_dir}"):
            global_hash_mp(source, files, data_dir, in_type, n_parallel, out_dir, is_norm)
        
    else:
        # sub process
        source = args.source
        proc_id = args.proc_id
        in_dir = args.in_dir
        out_dir = args.out_dir
        in_file_list = eval(args.file_list)
        in_type = args.in_type
        is_norm = args.is_norm
    
        out_type = 'parquet'
        file_args = [(os.path.join(in_dir, f_name), os.path.join(out_dir, f"{f_name}.id_hash.{out_type}"), f_name) for f_name in in_file_list]

        with Timer(f"generate hash index with proc-id {proc_id}"):
            generate_hash_index(proc_id, in_type, file_args, source, is_norm)