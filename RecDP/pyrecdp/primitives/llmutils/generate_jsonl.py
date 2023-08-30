import argparse
import os
from multiprocessing import Pool, cpu_count
from pyrecdp.core.utils import Timer
from math import ceil
from tqdm import tqdm
import json

def convert_to_json(x):
    in_file_name_list, out_file_name = x
    with open(out_file_name, 'w') as f:
        for in_file_name in in_file_name_list:            
            with open(in_file_name, 'r') as rdr:
                doc = rdr.read()
                record = {"text": doc, "meta": {'filename': os.path.basename(in_file_name)}}
                f.write(json.dumps(record) + "\n")                 
    return True

def convert_to_json_MP(data_dir, out_dir, n_part):
    os.makedirs(out_dir, exist_ok=True)
    files = list(sorted(os.listdir(data_dir)))
    files = [os.path.join(data_dir, i) for i in files]

    n_proc = cpu_count()
    if n_part < n_proc:
        n_proc = n_part
    n_chunks = ceil(len(files) / n_proc)
    remain = len(files) % n_proc
    if n_chunks == 1 and remain:
        n_proc = remain
    print(f"resetting to {n_proc} for number of processes")
    
    args = [(files[i : i + n_chunks], os.path.join(out_dir, f"part_{part_id}.jsonl")) for part_id, i in enumerate(range(0, len(files), n_chunks))]
    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(convert_to_json, args), total=len(args),
        )
        for test in pbar:
            pbar.update()
            if test:
                continue
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("-n", dest="n_part", type=int, default = 10)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    n_part = args.n_part
    
    with Timer(f"apply duplicates.pickle to create new data"):
        convert_to_json_MP(data_dir, out_dir, n_part)
    