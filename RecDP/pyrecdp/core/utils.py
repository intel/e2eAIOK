import numpy as np
import pandas as pd

import timeit

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
    
    type_change = False
    if not type_change:
        type_change = try_text(s)
        
    return type_change

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