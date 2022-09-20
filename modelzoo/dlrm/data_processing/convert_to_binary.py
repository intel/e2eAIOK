import logging
from timeit import default_timer as timer
import os, sys
import numpy as np
import pandas as pd
import subprocess
import argparse
from math import ceil
import shutil
from tqdm import tqdm

###############################################
# !!!put HDFS NODE here, empty won't proceed!!!
HDFS_NODE = "sr140"
###############################################

LABEL_COL = f"_c0"
INT_COLS = [f"_c{i}" for i in list(range(1, 14))]
CAT_COLS = [f"_c{i}" for i in list(range(14, 40))]
sorted_column_name = [f"_c{i}" for i in list(range(0, 40))]
path_prefix = f"hdfs://{HDFS_NODE}:9000"
output_folder = "/home/vmagent/app/dataset/criteo/output/"
current_path = "/home/vmagent/app/dataset/criteo/output/"

def _process_files(filename, is_train):
    f_name = filename.split('/')[-1]
    if path_prefix.startswith("hdfs://"):
        local_folder = os.getcwd()
        if not os.path.exists(f"{local_folder}/{f_name}"):
            if not is_train:
                print(f"Start to Download {filename} from HDFS to {local_folder}")
            process = subprocess.Popen(["/home/hadoop-3.3.1/bin/hdfs", "dfs", "-get", f"{filename}"])
            process.wait()
    else:
        local_folder = current_path
    if not is_train:
        print(f"Start to convert {local_folder}/{f_name} to numpy binary")
    pdf = pd.read_parquet(f"{local_folder}/{f_name}")
    pdf[LABEL_COL] = pdf[LABEL_COL].astype(np.int32)
    pdf[INT_COLS] = pdf[INT_COLS].astype(np.int32)
    pdf[CAT_COLS] = pdf[CAT_COLS].astype(np.int32)
    pdf = pdf[sorted_column_name]
    pdf= pdf.to_records(index=False)
    pdf= pdf.tobytes() 
    if path_prefix.startswith("hdfs://"):
        if not is_train:
            print(f"Remove downloaded {local_folder}/{f_name}")
        try:
            os.remove(f"{local_folder}/{f_name}")
        except:
            shutil.rmtree(f"{local_folder}/{f_name}")
    return pdf

def process_files(files, output_name, is_train = False):
    os.makedirs(output_folder, exist_ok=True)
    output_name = f"{output_folder}/{output_name}"
    if os.path.exists(output_name):
        print(f"{output_name} exists, skip this process")
        return
    with open(output_name, "wb") as wfd:
        for filename in files:
            f_name = filename.split('/')[-1]
            t11 = timer()
            pdf = _process_files(filename, is_train)
            wfd.write(pdf)
            t12 = timer()
            if not is_train:
                print(f"Convert {f_name} to binary completed, took {(t12 - t11)} secs")

def concat_days(input_name, output_name, show_progress = False):
    idx = len(input_name)
    output_name = f"{output_folder}/{output_name}"

    t1 = timer()
    #input_name = f"{current_path}/bin/train_data.bin"
    with open(output_name, "ab") as wfd:
        if show_progress:
            for part_i in tqdm(input_name, total=len(input_name)):
                part_i = f"{output_folder}/{part_i}"
                with open(f"{part_i}", 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                os.remove(f"{part_i}")
        else:
            for part_i in input_name:
                part_i = f"{output_folder}/{part_i}"
                with open(f"{part_i}", 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                os.remove(f"{part_i}")
    t2 = timer()
    if show_progress:
        print(f"Merged temp files to {output_name}, took " + "%.3f secs" % (t2 - t1))

def merge_days(files, output_name):
    output_name = f"{output_folder}/{output_name}"
    #print(f"Start merge to {output_name}")
    cache = []
    for filename in files:
        cache.append(_process_files(filename, True))
    # we will randomly choose one row from each file to merge
    # use round robin
    bytes_per_feature = 4
    tot_fea = 40
    bytes_per_row = (bytes_per_feature * tot_fea)
    
    sizes = [int(len(cache[part_i])/bytes_per_row) for part_i in range(len(cache))]
    #print(f"{output_name}: {sizes}")
    max_len = np.max(sizes)

    with open(output_name, "wb") as wfd:
        for i in range(0, max_len):
            start = i * bytes_per_row
            end = start + bytes_per_row
            for part_i in range(len(cache)):
                if i >= sizes[part_i]:
                    continue
                wfd.write(cache[part_i][start:end])


def post_process(process_type, mp):
    print(f"multi process is {mp}")
    t0 = timer()
    days = 23
    if process_type == "train":
        # get file lists
        output_name = "train_data.bin"
        if os.path.exists(f"{output_folder}/{output_name}"):
            print(f"{output_folder}/{output_name} exists, won't proceed")
            return
        parquet_files_list = []
        if path_prefix.startswith("hdfs://"):
            for day in range(days):
                print(f"List files in {path_prefix}/{current_path}/dlrm_categorified_day_{day}/")
                process = subprocess.Popen(["/home/hadoop-3.3.1/bin/hdfs", "dfs", "-ls", f"{path_prefix}/{current_path}/dlrm_categorified_day_{day}/"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, err = process.communicate()
                files = [i.split()[-1] for i in output.decode("utf-8").split('\n') if len(i) > 0]
                parquet_files = [f"{file_name}" for file_name in files if file_name.endswith('parquet')]
                parquet_files_list.append(parquet_files)
        else:
            for day in range(days):
                parquet_files = [f"{file_name}" for file_name in os.listdir(f"{current_path}/dlrm_categorified_day_{day}/") if file_name.endswith('parquet')]
                parquet_files_list.append(parquet_files)
        # now all parquet_files_list is divided by day
        total_num_batch = len(parquet_files_list[0])
        batch = [[parquet_files_list[j][i] for j in range(len(parquet_files_list))] for i in range(total_num_batch)]

        start = 0
        pool = []
        pool_output = []
        for start in tqdm(range(total_num_batch), total = total_num_batch, smoothing=0.0):
            #print(f"Create subprocess {idx} for {output_name}[{start}:{end}], total {num_part}")
            src_files_list = batch[start]
            p_output_name = f"{output_name}.{start}"
            pool_output.append(p_output_name)
            #print(" ".join(["python", "convert_to_binary.py", "-p", f"{src_files_list}", "-o", f"{p_output_name}"]))
            pool.append(subprocess.Popen(["python", "convert_to_binary.py", "-p", f"{src_files_list}", "-o", f"{p_output_name}"]))
            if len(pool) >= mp or (start + 1) == total_num_batch:
                for p in pool:
                    p.wait()
                pool = []
                # merge to final output
                concat_days(pool_output, output_name)
                pool_output = []
        t1 = timer()
        print(f"All subprocess for {output_name} completed, took " +  "%.3f secs" % (t1 - t0))
      
    elif process_type == "test":
        p_files = [f"{path_prefix}/{current_path}/dlrm_categorified_test"]
        output_name = "test_data.bin"
        process_files(p_files, output_name)
    else:
        p_files = [f"{path_prefix}/{current_path}/dlrm_categorified_valid"]
        output_name = "valid_data.bin"
        process_files(p_files, output_name)

    t_end = timer()
    print(f"Completed for {output_name}, took " +  "%.3f secs" % (t_end - t0))

def process_dicts():
    if os.path.exists(f"{output_folder}/day_fea_count.npz"):
        print(f"{output_folder}/day_fea_count.npz exists, skip")
        return
    print("Start to generate day_fea_count.npz")
    if path_prefix.startswith("hdfs://"):
        local_folder = os.getcwd()
        filename = f"dicts"
        print(f"Start to Download {filename} from HDFS to {local_folder}")
        process = subprocess.Popen(["/home/hadoop-3.3.1/bin/hdfs", "dfs", "-get", f"{path_prefix}/{current_path}/{filename}"])
        process.wait()
    feat_dim_final = []
    for name in CAT_COLS:
        print(f"Get dimension of {name}")
        c = pd.read_parquet(f"{local_folder}/dicts/{name}")
        feat_dim_final.append(c.shape[0])
    print(feat_dim_final)
    np.savez(open(f"{output_folder}/day_fea_count.npz", "wb"), counts=np.array(feat_dim_final))
    
def main(settings):
    if HDFS_NODE == "":
        print("Please add correct HDFS_NODE name in this file, or this script won't be able to process")
        return
    if settings.partitioned_file:
        p_files = eval(settings.partitioned_file)
        merge_days(p_files, settings.output_name)
    else:
        t1 = timer()
        #process_dicts()
        #post_process("test", 1)
        #post_process("valid", 1)
        post_process("train", int(settings.multi_process))
    
        t3 = timer()
        print(f"Total process time is {(t3 - t1)} secs")

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_name')
    parser.add_argument('-p', '--partitioned_file')
    parser.add_argument('-mp', '--multi_process', default=10)
    return parser.parse_args(args)
    
if __name__ == "__main__":
    input_args = parse_args(sys.argv[1:])
    main(input_args)
    
