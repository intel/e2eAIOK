import logging
from timeit import default_timer as timer
import os
import numpy as np
import pandas as pd

# Define Schema
LABEL_COL = 0
INT_COLS = [f"_c{i}" for i in list(range(1, 14))]
CAT_COLS = [f"_c{i}" for i in list(range(14, 40))]
sorted_column_name = [f"_c{i}" for i in list(range(1, 40))]

def list_dir(path):
    return [f"{path}/{file_name}" for file_name in os.listdir(path) if file_name.endswith('parquet')]

def make_partition(files, num_part):
    start = 0
    total = len(files)
    step = int(total / num_part)
    partitioned_files = []
    if total % num_part != 0:
        step = step + 1
    for i in range(num_part):
        end = start + step if (start + step) < total else total
        if start == end:
            break
        partitioned_files.append(files[start: end])
        start = end
    return partitioned_files


def process_convert(part_i, num_part, src_files, max_ind_range):
    feat_dims = [0] * len(CAT_COLS)
    print(f"Start to process {part_i + 1}/{num_part}")
    t1 = timer()
    pdf = pd.read_parquet(src_files)
    pdf = pdf[sorted_column_name]
    pdf[INT_COLS] = pdf[INT_COLS].astype(np.float32)
    pdf[CAT_COLS] = pdf[CAT_COLS] % max_ind_range
    for i, c_name in enumerate(CAT_COLS):
        c_max = pdf[c_name].max()
        if c_max > feat_dims[i]:
            feat_dims[i] = c_max
    pdf= pdf.to_records(index=False)
    pdf= pdf.tobytes()
    t3 = timer()
    print(f"{part_i + 1}/{num_part} completed, " + "%.3f secs" % (t3 - t1))
    return feat_dims, pdf


def post_process(input_name, output_name, expected_num_part = 1, max_ind_range = 40000000, create_dict = ""):
    # list files
    files = list_dir(input_name)
    # make partition
    partitioned_file = make_partition(files, expected_num_part)
    # convert
    num_part = len(partitioned_file)
    feat_dim_final = [0] * len(CAT_COLS)
    with open(output_name, "wb") as f:
        for part_i, src_files in enumerate(partitioned_file):
            feat_dim, pdf = process_convert(part_i, num_part, src_files, max_ind_range)
            f.write(pdf)
            for idx, v in enumerate(feat_dim):
                if feat_dim_final[idx] < v:
                    feat_dim_final[idx] = v
    print(feat_dim_final)
    if create_dict != "":
        np.savez(open(create_dict, "wb"), counts=np.array(feat_dim_final))

def main():
    current_path = "/home/vmagent/app/dataset/criteo/output/"

    train_input = "dlrm_categorified"
    test_input = "dlrm_categorified_test"
    valid_input = "dlrm_categorified_valid"
    post_process(f"{current_path}{train_input}", f"{current_path}/train_data.bin", 100, create_dict = f"{current_path}/day_fea_count.npz")
    post_process(f"{current_path}{test_input}", f"{current_path}/test_data.bin")
    post_process(f"{current_path}{valid_input}", f"{current_path}/valid_data.bin")


if __name__ == "__main__":
    main()
    