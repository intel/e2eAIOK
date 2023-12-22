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

import numpy as np
import pandas as pd
import copy
import os

import timeit
import pickle

# `def pretty(d, indent=0):
#     from pyrecdp.primitives.operations import Operation
#     for key, value in d.items():
#         print('\t' * indent + str(key))
#         if isinstance(value, dict):
#             pretty(value, indent+1)
#         elif isinstance(value, list):
#             for a in value:
#                 pretty(a)
#         elif isinstance(value, Operation):
#             print('\t' * (indent+1) + str(value))

def callable_string_fix(func_str):
    func_str_lines = func_str.split('\n')
    import re
    bad_indent = "".join(re.findall("^(\s+)", func_str_lines[0]))
    if len(bad_indent) == 0:
        return func_str
    func_str = '\n'.join([i[len(bad_indent):] for i in func_str_lines])
    return func_str
    
def sequenced_union1d(a, b):
    if isinstance(a, np.ndarray):
        a_list = a.tolist()
    elif isinstance(a, list):
        a_list = a
    if isinstance(b, np.ndarray):
        b_list = b.tolist()
    elif isinstance(b, list):
        b_list = b
    a_dict = dict((i, 0) for i in a_list)
    [a_list.append(i) for i in b_list if i not in a_dict]
    return np.array(a_list)

def get_encoder(dict_path):
    if dict_path.endswith(".parquet"):
        return get_encoder_df(dict_path)
    else:
        if not os.path.exists(dict_path):
            return FileNotFoundError(f"{dict_path} is not found")
        with open(dict_path, 'rb') as f:
            encoder = pickle.load(f)
        return encoder

def save_encoder(encoder, dict_path):
    if isinstance(encoder, pd.DataFrame):
        return save_encoder_df(encoder, dict_path)
    else:
        dirname = os.path.dirname(dict_path)
        if len(dirname) > 0 and not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(dict_path, 'wb') as f:
            pickle.dump(encoder, f)

def get_encoder_df(dict_path):
    if not os.path.exists(dict_path):
        return FileNotFoundError(f"{dict_path} is not found")
    return pd.read_parquet(dict_path)

def save_encoder_df(encoder, dict_path):
    if isinstance(dict_path, type(None)):
        return
    dirname = os.path.dirname(dict_path)
    if len(dirname) > 0 and not os.path.exists(dirname):
        os.mkdir(dirname)
    encoder.to_parquet(dict_path)
    
def deepcopy(dict_df):
    return copy.deepcopy(dict_df)

def fillna_with_series(tgt_s, src_s):
    nan_loc = tgt_s.isnull().to_list()
    df_encoded_list = tgt_s.to_list()
    df_encoded_2_list = src_s.to_list()

    ret = []
    for idx, is_nan in enumerate(nan_loc):
        if not is_nan:
            ret.append(df_encoded_list[idx])
        else:
            ret.append(df_encoded_2_list[idx])
    
    ret = pd.Series(ret, index = tgt_s.index)
    return ret

def get_sample_indices_pd(indices, target_num_rows):
    if isinstance(indices, pd.DataFrame):
        indices = indices.notna()
    if indices.size > target_num_rows:
        frac = target_num_rows/indices.size
        mask = pd.Series(np.random.choice(a=[False, True], size=(indices.size), p=[1-frac, frac]))
        indices = indices & mask
    return indices

def sample_read(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, nrows = 10000)
    elif file_path.endswith('.parquet'):
        from pyarrow.parquet import ParquetFile
        import pyarrow as pa
        pf = ParquetFile(file_path) 
        sample_rows = next(pf.iter_batches(batch_size = 10000)) 
        return pa.Table.from_batches([sample_rows]).to_pandas() 
    else:
        raise NotImplementedError("now sample read only support csv and parquet")

def dump_fix(x, base_dir):
    import json
    import cloudpickle
    import hashlib
    import os
    import inspect

    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = dump_fix(v, base_dir)
    elif isinstance(x, list) or isinstance(x, np.ndarray):
        for idx in range(len(x)):
            x[idx] = dump_fix(x[idx], base_dir)
    elif isinstance(x, type):
        x = (x.__module__, x.__name__)
    elif isinstance(x, tuple):
        x = [dump_fix(i, base_dir) for i in x]
    elif hasattr(x, 'mydump'):
        x = x.mydump()
    elif callable(x):
        func_str = callable_string_fix(inspect.getsource(x))
        #print(func_str)
        md5 = hashlib.md5(func_str.encode('utf-8')).hexdigest()
        uuid_name = f"{md5}.bin"
        uuid_name = os.path.join(base_dir, uuid_name)
        with open(uuid_name, 'wb') as f:
            ret = cloudpickle.dumps(x)
            f.write(ret)
        x = uuid_name
    else:
        try:
            json.dumps(x)
            x = x
        except:
            x = str(x)
    return x

def class_name_fix(s):
    ret = s
    if isinstance(s, type):
        ret = s
    elif isinstance(s, tuple) or isinstance(s, list):
        import importlib
        module = importlib.import_module(s[0])
        ret = eval("module." + s[1])
    return ret

def infer_problem_type(df, label):
    if label is None:
        return None
    if isinstance(df, str):
        if df.endswith('.csv'):
            y = pd.read_csv(df)[label]
        elif df.endswith('.parquet'):
            y = pd.read_parquet(df)[label]
        else:
            raise NotImplementedError("Load DataFrame based on path only support csv and parquet")
    elif isinstance(df, pd.DataFrame):
        y = df[label]
    unique_count = y.nunique()
    if unique_count == 2:
        problem_type = 'binary'
    elif y.dtype.name in ['object', 'category', 'string']:
        problem_type = 'multiclass'
    else:
        problem_type = 'regression'
    return problem_type
 
def is_text_series(s):
    from pandas.api import types as pdt
    import pandas as pd
    def try_text(s):
        if not pdt.is_string_dtype(s):
            return False
        if len(s) > 1000:
            # Sample to speed-up type inference
            result = s.sample(n=1000, random_state=0)
        try:
            avg_words = pd.Series(result).str.split().str.len().mean()
            if avg_words > 2:
                return True
        except:
            return False
        return False
    
    return try_text(s)

def is_tuple(s):
    if isinstance(s.first_valid_index(), type(None)):
        t = None
    else:
        t = type(s.loc[s.first_valid_index()]) if s.first_valid_index() >= 0 else None
    return isinstance(t, tuple)
       
def is_integer_convertable(s):
    from pandas.api.types import is_numeric_dtype
    if not is_numeric_dtype(s.dtype):
        return False
    s = s.fillna(0)
    if np.array_equal(s, s.astype(int)):
        return True
    return False

def is_unique(s):
    if isinstance(s, pd.Series):
        return len(s.unique()) == 1
    elif isinstance(s, pd.DataFrame) and len(s.columns) == 1:
        return len(s[s.columns[0]].unique()) == 1
    else:
        return False

class Timer:
    level = 0
    viewer = None

    def __init__(self, name):
        self.name = name
        if Timer.viewer:
            Timer.viewer.display(f"{name} started ...")
        else:
            print(f"{name} started ...")

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if Timer.viewer:
            Timer.viewer.display(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        else:
            print(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def update_linklist(linklist, key, value):
    if key not in linklist:
        linklist[key] = []
    linklist[key].append(value)
    return linklist

