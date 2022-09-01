import logging
from math import ceil
import shutil
from timeit import default_timer as timer
import os, sys
import numpy as np
import pandas as pd
import argparse
import subprocess

# Define Schema
LABEL_COL = 0
INT_COLS = [f"_c{i}" for i in list(range(1, 14))]
CAT_COLS = [f"_c{i}" for i in list(range(14, 40))]
sorted_column_name = [f"_c{i}" for i in list(range(1, 40))]

def list_dir(path):
    return [f"{file_name}" for file_name in os.listdir(path) if file_name.endswith('parquet')]

def make_partition(files, num_part):
    start = 0
    total = len(files)
    step = ceil(total / num_part)
    partitioned_files = []
    for i in range(num_part):
        end = start + step if (start + step) < total else total
        if start == end:
            break
        partitioned_files.append(files[start: end])
        start = end
    return partitioned_files


def process_convert(output_name, part_i, num_part, src_files, max_ind_range):
    feat_dims = [0] * len(CAT_COLS)
    print(f"{output_name} processing {part_i + 1}/{num_part}")
    t1 = timer()
    pdf = pd.read_parquet(src_files)
    pdf = pdf[sorted_column_name]
    pdf[INT_COLS] = pdf[INT_COLS].fillna(0).astype(np.float32)
    pdf[CAT_COLS] = pdf[CAT_COLS] % max_ind_range
    for i, c_name in enumerate(CAT_COLS):
        c_max = pdf[c_name].max()
        if c_max > feat_dims[i]:
            feat_dims[i] = c_max
    #print(pdf)
    pdf= pdf.to_records(index=False)
    pdf= pdf.tobytes()
    t3 = timer()
    print(f"{output_name} completed {part_i + 1}/{num_part}, took " + "%.3f secs" % (t3 - t1))
    return feat_dims, pdf


def partial_post_process(partitioned_file, output_name, max_ind_range):
    feat_dim_final = [0] * len(CAT_COLS)
    with open(output_name, "wb") as f:
        num_part = len(partitioned_file)
        for part_i, src_files in enumerate(partitioned_file):
            feat_dim, pdf = process_convert(output_name, part_i, num_part, src_files, max_ind_range)
            f.write(pdf)
            for idx, v in enumerate(feat_dim):
                if feat_dim_final[idx] < v:
                    feat_dim_final[idx] = v
    np.savez(open(f"{output_name}_feat.npz", "wb"), counts=np.array(feat_dim_final))


def post_process(input_name, output_name, num_parallel = 1, expected_num_part = 1, create_dict = ""):
    t0 = timer()
    # 1. list files
    files = list_dir(input_name)
    # 2. make partition
    num_parallel = num_parallel if num_parallel < expected_num_part else expected_num_part
    partitioned_file = make_partition(files, expected_num_part)
    # 3. convert
    num_part = len(partitioned_file)
    # 3.1 create partial output
    start = 0
    end = start
    step = ceil(num_part / num_parallel)
    idx = 0
    pool = []
    while end != num_part:
        end = start + step if (start + step) < num_part else num_part
        print(f"Create subprocess {idx} for {output_name}[{start}:{end}], total {num_part}")
        src_files_list = partitioned_file[start:end]
        p_output_name = f"{output_name}_{idx}"
        pool.append(subprocess.Popen(["python", "splitconversion.py", "-d", f"{input_name}", "-p", f"{src_files_list}", "-o", f"{p_output_name}"]))
        idx += 1
        start = end
    for p in pool:
        p.wait()
    t1 = timer()
    print(f"All subprocess for {output_name} completed, took " +  "%.3f secs" % (t1 - t0))

    # 3.2 merge partial output
    print(f"Start Final Merge for {output_name}")
    feat_dim_final = [0] * len(CAT_COLS)
    with open(output_name, "wb") as wfd:
        for part_i in range(idx):
            t1 = timer()
            with open(f"{output_name}_{part_i}", 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
            os.remove(f"{output_name}_{part_i}")
            t2 = timer()
            print(f"Done copy {output_name}_{part_i}/{idx} file, took " + "%.3f secs" % (t2 - t1))
            with open(f"{output_name}_{part_i}_feat.npz", 'rb') as pnpz:
                feat_dim = np.load(pnpz)['counts']
            for i, v in enumerate(feat_dim):
                if feat_dim_final[i] < v:
                    feat_dim_final[i] = v
            os.remove(f"{output_name}_{part_i}_feat.npz")
    if create_dict != "":
        np.savez(open(create_dict, "wb"), counts=np.array(feat_dim_final))
    t_end = timer()
    print(f"Completed for {output_name}, took " +  "%.3f secs" % (t_end - t0))
    print(feat_dim_final)


def main(settings):
    current_path = "/home/vmagent/app/dataset/criteo/output/"
    if settings.partitioned_file:
        p_files = [[f"{settings.dir}/{j}" for j in i] for i in eval(settings.partitioned_file)]
        partial_post_process(p_files, settings.output_name, settings.max_ind_range)
    else:
        train_input = "dlrm_categorified"
        test_input = "dlrm_categorified_test"
        valid_input = "dlrm_categorified_valid"
        post_process(f"{current_path}{train_input}", f"{current_path}/train_data.bin", expected_num_part = 100, num_parallel = settings.multi_process, create_dict = f"{current_path}/day_fea_count.npz")
        post_process(f"{current_path}{test_input}", f"{current_path}/test_data.bin", expected_num_part = 10, num_parallel = settings.multi_process)
        post_process(f"{current_path}{valid_input}", f"{current_path}/valid_data.bin", expected_num_part = 10, num_parallel = settings.multi_process)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-o', '--output_name')
    parser.add_argument('-r', '--max_ind_range', default = 40000000, type = int)
    parser.add_argument('-p', '--partitioned_file')
    parser.add_argument('-mp', '--multi_process', default = 15, type = int)
    return parser.parse_args(args)
    
if __name__ == "__main__":
    input_args = parse_args(sys.argv[1:])
    main(input_args)    