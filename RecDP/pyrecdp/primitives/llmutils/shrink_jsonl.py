import argparse
import os
import pickle
from pyrecdp.core.utils import Timer
from .utils import get_nchunks_and_nproc, launch_mp

def shrink_document(x_list):
    for x in x_list:
        in_file_name, out_file_name, dedup_list = x
        with open(in_file_name, 'r') as rdr:
            with open(out_file_name, 'w') as f:
                for idx, line in enumerate(rdr):
                    if idx not in dedup_list:
                        f.write(line)
    return True

def shrink_document_MP(data_dir, dup_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pickle_path = dup_dict

    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    
    dup_dict = pickle.load(open(pickle_path, 'rb'))
    shink_args = [(os.path.join(data_dir, i), os.path.join(out_dir, i), dup_dict[i]) for i in files]

    n_chunks, n_proc = get_nchunks_and_nproc(len(files))
    print(f"resetting to {n_proc} for number of processes")
    
    shink_args = [shink_args[i : i + n_chunks] for i in range(0, len(shink_args), n_chunks)]
    launch_mp(n_proc, shink_args, shrink_document)    
