"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
import os, sys
from pyrecdp.core.utils import Timer
import json
from pyrecdp.primitives.llmutils.utils import MultiProcessManager, get_target_file_list, get_nchunks_and_nproc
import pandas as pd

def process_text_to_parquet(in_file_name_list, out_file_name):
    text_list = []
    meta_list = []
    pdf = pd.DataFrame()
    for in_file_name in in_file_name_list:            
        with open(in_file_name, 'r') as rdr:
            doc = rdr.read()
            text_list.append(doc)                
            meta_list.append({'filename': os.path.basename(in_file_name)})           
    pdf['text'] = pd.Series(text_list)
    pdf['meta'] = pd.Series(meta_list)
    pdf.to_parquet(out_file_name)

             
def process_csv_to_parquet(in_file_name, out_file_name):
    pdf = pd.read_csv(in_file_name).reset_index(drop=True)
    key = 'text' if 'text' in pdf.columns else 'content'
    if key != 'text':
        pdf.rename(columns={key: 'text'})

    pdf.to_parquet(out_file_name)
    del pdf
    
def process_jsonl_to_parquet(in_file_name, out_file_name):
    text_list = []
    meta_list = []
    pdf = pd.DataFrame()
    with open(in_file_name, 'r') as rdr:         
        for idx, line in enumerate(rdr):
            ob = json.loads(line)
            text = ob['text']
            if 'meta' not in ob:
                ob['meta'] = str({})
            text_list.append(text)
            meta_list.append(ob['meta'])
    pdf['text'] = pd.Series(text_list)
    pdf['meta'] = pd.Series(meta_list)
    pdf.to_parquet(out_file_name)


def convert_impl_mp(proc_id, in_type, x_list):
    for x in x_list:
        try:
            in_file_name, out_file_name, base_file_name = x
            base_file_name = os.path.basename(base_file_name)
            out_dir = os.path.dirname(out_file_name)
            os.makedirs(out_dir, exist_ok=True)
            if in_type == 'parquet':
                process_csv_to_parquet(in_file_name, out_file_name)
            elif in_type == 'jsonl':
                process_jsonl_to_parquet(in_file_name, out_file_name)
            elif in_type == 'text':
                process_text_to_parquet(in_file_name, out_file_name)
            else:
                raise NotImplementedError(f"Unsupported file type {in_type}")
        except Exception as e:
            with open(f"{out_file_name}.error.log", 'w') as f:
                f.write(f"Failed to process {base_file_name}, error is {e}")
    return True

def convert(data_dir, in_type, n_parallel, out_dir):
    files = get_target_file_list(data_dir, in_type if in_type != 'text' else '*')
    n_chunks, n_proc = get_nchunks_and_nproc(len(files), n_part = n_parallel)
    print(f"resetting to {n_proc} for number of processes")
    
    if in_type != 'text':
        args = [(idx, [i]) for idx, i in enumerate(files)]
    else:
        args = [(part_id, files[i : i + n_chunks]) for part_id, i in enumerate(range(0, len(files), n_chunks))]    
    
    cmdline_args = []
    for (idx, arg) in args:
        cmd = ["--proc_id", f"{idx}", "--in_dir", f"{data_dir}", "--out_dir", f"{out_dir}", "--in_type", f"{in_type}", "--file_list", f"{arg}"]
        cmdline_args.append((idx, cmd))

    mp_mgr = MultiProcessManager()
    mp_mgr.launch_cmdline_mp(cmdline_args, n_proc, f"{__file__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_id", dest="proc_id", type=int, default = -1)
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("--in_type", dest="in_type", type=str, default="jsonl")
    parser.add_argument("--in_dir", dest="in_dir", type=str)
    parser.add_argument("--out_dir", dest="out_dir", type=str)
    parser.add_argument("--file_list", dest="file_list", type=str)
    parser.add_argument("-mp", dest="mp", type=int, default=-1)
    args = parser.parse_args()
    
    # main controller
    if args.proc_id == -1:
        data_dir = args.data_dir
        out_dir = args.out_dir
        in_type = args.in_type
        n_parallel = args.mp
        is_norm = args.is_norm
        print(f"is_norm is {is_norm}")
        files = get_target_file_list(data_dir, in_type if in_type != 'text' else '')
        if len(files) == 0:
            print("Detect 0 files, exit here")
            sys.exit(0)
            
        with Timer(f"generate hash to {data_dir}"):
            convert(files, data_dir, in_type, n_parallel, out_dir)
    else:
        # sub process
        proc_id = args.proc_id
        in_dir = args.in_dir
        out_dir = args.out_dir
        in_file_list = eval(args.file_list)
        in_type = args.in_type
    
        out_type = 'parquet'
        if in_type != 'text':
            file_args = [(os.path.join(in_dir, f_name), os.path.join(out_dir, f"{f_name}.{out_type}"), f_name) for f_name in in_file_list]
            with Timer(f"generate hash index with proc-id {proc_id}"):
                convert_impl_mp(proc_id, in_type, file_args)
        else:
            file_args = [([os.path.join(in_dir, f_name) for f_name in in_file_list], os.path.join(out_dir, f"part_{proc_id}.parquet"), in_dir)]
            with Timer(f"generate hash index with proc-id {proc_id}"):
                convert_impl_mp(proc_id, in_type, file_args)