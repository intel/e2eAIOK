import logging
from timeit import default_timer as timer
import os, sys
import numpy as np
import pandas as pd
import subprocess #nosec
import argparse
from math import ceil
import shutil

###############################################
# !!!put HDFS NODE here, empty won't proceed!!!
HDFS_NODE = ""
###############################################

LABEL_COL = f"_c0"
INT_COLS = [f"_c{i}" for i in list(range(1, 14))]
CAT_COLS = [f"_c{i}" for i in list(range(14, 40))]
sorted_column_name = [f"_c{i}" for i in list(range(0, 40))]

def process(files, output_name, path_prefix, output_folder, current_path):
    os.makedirs(output_folder, exist_ok=True)
    output_name = f"{output_folder}/{output_name}"
    if os.path.exists(output_name):
        print(f"{output_name} exists, skip this process")
        return
    with open(output_name, "wb") as wfd:
        for filename in files:
            t11 = timer()
            if path_prefix.startswith("hdfs://"):
                local_folder = os.getcwd()
                if not os.path.exists(f"{local_folder}/{filename}"):
                    print(f"Start to Download {filename} from HDFS to {local_folder}")
                    process = subprocess.Popen(["/home/hadoop-3.3.1/bin/hdfs", "dfs", "-get", f"{path_prefix}/{current_path}/{filename}"])
                    process.wait()
            else:
                local_folder = current_path
            print(f"Start to convert parquet files to numpy binary")
            pdf = pd.read_parquet(f"{local_folder}/{filename}")
            pdf[LABEL_COL] = pdf[LABEL_COL].astype(np.int32)
            pdf[INT_COLS] = pdf[INT_COLS].astype(np.int32)
            pdf[CAT_COLS] = pdf[CAT_COLS].astype(np.int32)
            # pdf[LABEL_COL] = pdf[LABEL_COL].fillna(0).astype(np.int32)
            # pdf[INT_COLS] = pdf[INT_COLS].fillna(0).astype(np.int32)
            # pdf[CAT_COLS] = pdf[CAT_COLS].fillna(0).astype(np.int32)
            pdf = pdf[sorted_column_name]
            pdf= pdf.to_records(index=False)
            pdf= pdf.tobytes()
            print(f"Start to write binary to {output_name}")
            wfd.write(pdf)
            if path_prefix.startswith("hdfs://"):
                print(f"Remove downloaded {local_folder}/{filename}")
                shutil.rmtree(f"{local_folder}/{filename}")
            t12 = timer()
            print(f"Convert {filename} to binary completed, took {(t12 - t11)} secs")

def concat_days(input_name, output_name, idx):
    print(f"Start Final Merge for {output_name}")
    #input_name = f"{current_path}/bin/train_data.bin"
    with open(output_name, "wb") as wfd:
        for part_i in range(idx):
            t1 = timer()
            with open(f"{input_name}.{part_i}", 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
            #os.remove(f"{output_name}.{part_i}")
            t2 = timer()
            print(f"Done copy {input_name}.{part_i} file, took " + "%.3f secs" % (t2 - t1))

def merge_days(input_name, output_name, idx):
    print(f"Start Final Merge for {output_name}")
    # we will randomly choose one row from each file to merge
    # use round robin
    bytes_per_feature = 4
    tot_fea = 40
    bytes_per_row = (bytes_per_feature * tot_fea)
    cache_num_row = 262144 * 40
    cur_row = 0
    
    sizes = [int(int(os.stat(f"{input_name}.{part_i}").st_size)/bytes_per_row) for part_i in range(idx)]
    print(sizes)
    max_len = np.max(sizes)
    opened_fd = [open(f"{input_name}.{part_i}", 'rb') for part_i in range(idx)]
    target_row = max_len
    total_round = ceil(target_row / cache_num_row)
    stop = False

    def ranout(cache):
        for c in cache:
            if c != None:
                return False
        return True

    with open(output_name, "wb") as wfd:
        cache = [None for _ in range(idx)]
        round_i = 0
        while not stop:
            t1 = timer()
            print("Loading cache ...")
            max_to_read_row = 0
            for part_i in range(idx):
                to_read_row = min(cache_num_row, sizes[part_i] - cur_row)
                max_to_read_row = max(max_to_read_row, to_read_row)
                if to_read_row > 0:
                    print(f"day_{part_i} to_read_row is {to_read_row}")
                    cache[part_i] = opened_fd[part_i].read(bytes_per_row * to_read_row)
                else:
                    cache[part_i] = None
            if ranout(cache):
                stop = True
                break
            max_to_read_row = min(max_to_read_row, cache_num_row)
            if max_to_read_row < cache_num_row:
                print("Last round of copy")
            end_cache_row = cur_row + max_to_read_row
            print(f"Writing to {output_name} {idx*cur_row/262144} to {idx*end_cache_row/262144} ...")
            for i in range(cur_row, end_cache_row):
                start = (i - cur_row) * bytes_per_row
                end = start + bytes_per_row
                for part_i in range(idx):
                    if cache[part_i] == None:
                        continue
                    if i >= sizes[part_i]:
                        continue
                    wfd.write(cache[part_i][start:end])
            t2 = timer()
            round_i += 1
            cur_row = end_cache_row
            print(f"Done copy {round_i}/{total_round} from all file, took " + "%.3f secs" % (t2 - t1))
            if end_cache_row >= target_row:
                stop = True

def post_process(process_type, mp, total_days, path_prefix, output_folder, current_path, settings):
    print(f"multi process is {mp}")
    t0 = timer()
    if process_type == "train":
        # 1 merge partial output
        num_part = total_days
        # 3.1 create partial output
        start = 0
        end = start
        step = 1
        idx = start
        pool = []
        train_files = [f"dlrm_categorified_day_{i}" for i in range(0, (num_part + 1))]
        output_name = "train_data.bin"
        while end != num_part:
            end = start + step
            print(f"Create subprocess {idx} for {output_name}[{start}:{end}], total {num_part}")
            src_files_list = train_files[start:end]
            if settings.local_small:
                p_output_name = f"{output_name}"
            else:
                p_output_name = f"{output_name}.{idx}"
            cmd = ["python", "convert_to_binary.py", "-p", f"{src_files_list}", "-o", f"{p_output_name}", "--dataset_path", settings.dataset_path]
            if settings.local_small:
                cmd += ['--local_small']
            pool.append(subprocess.Popen(cmd))
            idx += 1
            start = end
            if len(pool) >= mp or end == num_part:
                for p in pool:
                    p.wait()
                pool = []
        t1 = timer()
        print(f"All subprocess for {output_name} completed, took " +  "%.3f secs" % (t1 - t0))
      
    elif process_type == "test":
        p_files = ['dlrm_categorified_test']
        output_name = "test_data.bin"
        process(p_files, output_name, path_prefix, output_folder, current_path)
    else:
        p_files = ["dlrm_categorified_valid"]
        output_name = "valid_data.bin"
        process(p_files, output_name, path_prefix, output_folder, current_path)

    t_end = timer()
    print(f"Completed for {output_name}, took " +  "%.3f secs" % (t_end - t0))

def process_dicts(path_prefix, output_folder, current_path):
    print("Start to generate day_fea_count.npz")
    if path_prefix.startswith("hdfs://"):
        local_folder = os.getcwd()
        filename = f"dicts"
        print(f"Start to Download {filename} from HDFS to {local_folder}")
        process = subprocess.Popen(["/home/hadoop-3.3.1/bin/hdfs", "dfs", "-get", f"{path_prefix}/{current_path}/{filename}"])
        process.wait()
    else:
        local_folder = f"{current_path}"
    feat_dim_final = []
    for name in CAT_COLS:
        print(f"Get dimension of {name}")
        c = pd.read_parquet(f"{local_folder}/dicts/{name}")
        feat_dim_final.append(c.shape[0])
    print(feat_dim_final)
    np.savez(open(f"{output_folder}/day_fea_count.npz", "wb"), counts=np.array(feat_dim_final))
    
def main(settings):
    if settings.local_small:
        path_prefix = "file://"
        total_days = 1
    else:
        path_prefix = f"hdfs://{HDFS_NODE}:9000"
        total_days = 23
    output_folder = f"{settings.dataset_path}/output"
    current_path = f"{settings.dataset_path}/output"
    if settings.partitioned_file:
        p_files = eval(settings.partitioned_file)
        process(p_files, settings.output_name, path_prefix, output_folder, current_path)
    else:
        t1 = timer()
        process_dicts(path_prefix, output_folder, current_path)

        print("Start to convert train/valid/test")
        post_process("train", int(settings.multi_process), total_days, path_prefix, output_folder, current_path, settings)
        post_process("test", 1, total_days, path_prefix, output_folder, current_path, settings)
        post_process("valid", 1, total_days, path_prefix, output_folder, current_path, settings)

        input_name = f"{output_folder}/train_data.bin"
        output_name = f"{current_path}/train_data.bin"
        if total_days > 1:
            merge_days(input_name, output_name, total_days)
    
        t3 = timer()
        print(f"Total process time is {(t3 - t1)} secs")

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_name')
    parser.add_argument('-p', '--partitioned_file')
    parser.add_argument('-mp', '--multi_process', default=6)
    parser.add_argument('-dp', '--dataset_path',type=str,default="/home/vmagent/app/dataset/criteo",help='dataset path for criteo')
    parser.add_argument('--local_small', action='store_true', help='worker host list')
    return parser.parse_args(args)
    
if __name__ == "__main__":
    input_args = parse_args(sys.argv[1:])
    main(input_args)