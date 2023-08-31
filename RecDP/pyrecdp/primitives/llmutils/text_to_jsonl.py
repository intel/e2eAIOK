import argparse
import os
from pyrecdp.core.utils import Timer
import json
from .utils import get_nchunks_and_nproc, launch_mp

def text_to_jsonl(x):
    in_file_name_list, out_file_name = x
    with open(out_file_name, 'w') as f:
        for in_file_name in in_file_name_list:            
            with open(in_file_name, 'r') as rdr:
                doc = rdr.read()
                record = {"text": doc, "meta": {'filename': os.path.basename(in_file_name)}}
                f.write(json.dumps(record) + "\n")                 
    return True

def text_to_jsonl_MP(data_dir, out_dir, n_part):
    os.makedirs(out_dir, exist_ok=True)
    files = list(sorted(os.listdir(data_dir)))
    files = [os.path.join(data_dir, i) for i in files]

    n_chunks, n_proc = get_nchunks_and_nproc(len(files), n_part)
    print(f"resetting to {n_proc} for number of processes")
    
    args = [(files[i : i + n_chunks], os.path.join(out_dir, f"part_{part_id}.jsonl")) for part_id, i in enumerate(range(0, len(files), n_chunks))]
    launch_mp(n_proc, args, text_to_jsonl)