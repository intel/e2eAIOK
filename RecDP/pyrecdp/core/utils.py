import numpy as np
import pandas as pd

import timeit

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

def dump_fix(x):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = dump_fix(v)
    elif isinstance(x, list):
        for idx in range(len(x)):
            x[idx] = dump_fix(x[idx])
    elif isinstance(x, type):
        x = (x.__module__, x.__name__)
    elif isinstance(x, tuple):
        x = (dump_fix(x[0]), dump_fix(x[1]))
    elif hasattr(x, 'dump'):
        x = x.dump()
    else:
        x = x
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
    t = type(s.loc[s.first_valid_index()]) if s.first_valid_index() >= 0 else None
    return isinstance(t, tuple)
           
def is_encoded(s):
    line_id = s.first_valid_index()
    if line_id < 0:
        return False
    sample_data = s.loc[line_id:(line_id+1000)]
    from pyrecdp.primitives.generators.nlp import BertTokenizerDecode
    proc_ = BertTokenizerDecode().get_function()
    try:
        proc_(sample_data)
    except Exception as e:
        #print(e)
        return False
    return True
 
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