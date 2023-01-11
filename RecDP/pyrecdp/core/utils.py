import numpy as np
import pandas as pd

def get_sample_indices_pd(indices, target_num_rows):
    if isinstance(indices, pd.DataFrame):
        indices = indices.notna()
    if indices.size > target_num_rows:
        frac = target_num_rows/indices.size
        mask = pd.Series(np.random.choice(a=[False, True], size=(indices.size), p=[1-frac, frac]))
        indices = indices & mask
    return indices

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