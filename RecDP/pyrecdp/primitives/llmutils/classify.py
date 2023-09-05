import argparse
import os
from pyrecdp.core.utils import Timer
import json
from .utils import get_nchunks_and_nproc, launch_mp

## Place Holder, not done yet
# define actual work
def classify(x_list, classify_condition):
    raise NotImplementedError("This function is not completed yet, just place holder")
    for x in x_list:
        in_file_name, out_file_name = x
        with open(in_file_name, 'r') as rdr:
            with open(out_file_name, 'w') as f:
                for idx, line in enumerate(rdr):
                    if classify_condition(idx):
                        f.write(line + "\n")
    return True

# define how to do parallel here
def classify_MP(data_dir, classify_condition, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    
    args = [(os.path.join(data_dir, i), os.path.join(out_dir, i)) for i in files]

    n_chunks, n_proc = get_nchunks_and_nproc(len(files))
    print(f"resetting to {n_proc} for number of processes")
    
    args = [(args[i : i + n_chunks], classify_condition) for i in range(0, len(args), n_chunks)]
    launch_mp(n_proc, args, classify)