import argparse
import os
import sys
import pickle
import queue
from multiprocessing import Pool, cpu_count
from pyrecdp.core.utils import Timer
from math import ceil
from tqdm import tqdm

def shink_document(x_list):
    for x in x_list:
        in_file_name, out_file_name, dedup_list = x
        with open(in_file_name, 'r') as rdr:
            with open(out_file_name, 'w') as f:
                for idx, line in enumerate(rdr):
                    if idx not in dedup_list:
                        f.write(line + "\n")
    return True

def shink_document_MP(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir
    pickle_path = args.dup_dict
    files = sorted(os.listdir(args.data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    dup_dict = pickle.load(open(pickle_path, 'rb'))
    shink_args = [(os.path.join(args.data_dir, i), os.path.join(out_dir, i), dup_dict[i]) for i in files]

    n_proc = cpu_count()
    n_chunks = ceil(len(files) / n_proc)
    remain = len(files) % n_proc
    if n_chunks == 1 and remain:
        n_proc = remain
    print(f"resetting to {n_proc} for number of processes")
    
    shink_args = [shink_args[i : i + n_chunks] for i in range(0, len(shink_args), n_chunks)]
    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(shink_document, shink_args), total=len(shink_args),
        )
        for test in pbar:
            pbar.update()
            if test:
                continue
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_files, dup_dir, ngram_size, num_perm, bands, ranges
    #pipeline = minHashLSH_prepare(df, num_perm = 256, ngram_size = 6, bands = 9, ranges = 13)
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-f", dest="dup_dict", type=str, default=None)
    parser.add_argument("-o", dest="out_dir", type=str, default=None)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    dup_dir = os.path.join(data_dir, "deduplicate")
    if args.dup_dict is None:
        dup_dict = os.path.join(dup_dir, "duplicates.pickle")
    else:
        dup_dict = args.dup_dict
        
    if args.out_dir is None:
        out_dir = os.path.join(dup_dir, "output")
    else:
        out_dir = args.out_dir
    
    dedup_args = argparse.Namespace()
    dedup_args.data_dir = data_dir
    dedup_args.out_dir = out_dir
    dedup_args.dup_dict = dup_dict
    
    with Timer(f"apply duplicates.pickle to create new data"):
        shink_document_MP(dedup_args)
    