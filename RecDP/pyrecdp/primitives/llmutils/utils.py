import os
from multiprocessing import Pool, cpu_count
from math import ceil
from tqdm import tqdm
import ftfy
import string
import re

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
