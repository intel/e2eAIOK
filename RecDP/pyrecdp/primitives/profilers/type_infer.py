from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import pandas as pd
import numpy as np
from pandas.api import types as pdt
import copy
from featuretools.primitives.base import TransformPrimitive
from tqdm import tqdm

def try_category(s):
    if pdt.is_categorical_dtype(s) and not pdt.is_bool_dtype(s):
        return False
    n_unique = s.nunique()
    total_len = len(s)
    threshold = (total_len / 5)
    if 1 <= n_unique <= threshold:
        return True     
    return False

def try_onehot(s):
    n_unique = s.nunique()
    if n_unique < 10:
        return s.unique().tolist()
    return False

def try_datetime(s):
    if pdt.is_datetime64_any_dtype(s):
        return False
    if not pdt.is_string_dtype(s):
        return False
    if s.isnull().all():
        return False
    try:
        if len(s) > 500:
            # Sample to speed-up type inference
            result = s.sample(n=500, random_state=0)
        result = pd.to_datetime(result, errors='coerce')
        if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
            return False
        else:
            return True
    except:
        return False

def try_re_numeric(s):
    if s.isnull().all():
        return True
    try:
        if len(s) > 500:
            # Sample to speed-up type inference
            result = s.sample(n=500, random_state=0)
        import re
        result = result.apply(lambda x: "".join(re.findall(r'\d+.*\d*', x)))
        result = pd.to_numeric(result, errors='coerce')
        if result.isnull().mean() <= 0.8:  # If over 80% of the rows are NaN
            return True
        return False
    except:
        return False
     
def try_numeric(s):
    if pdt.is_numeric_dtype(s):
        return True
    if s.isnull().all():
        return True
    try:
        if len(s) > 500:
            # Sample to speed-up type inference
            result = s.sample(n=500, random_state=0)
        result = pd.to_numeric(result, errors='coerce')
        if result.isnull().mean() <= 0.8:  # If over 80% of the rows are NaN
            return True
        return False
    except:
        return False

def try_list_string(s):
    if not pdt.is_string_dtype(s):
        return False
    result = s[s.notna()]
    try:
        result = pd.Series(s).str.split(';')
        avg_words = result.str.len().mean()
        if avg_words > 1:
            estimated_size = result.explode().nunique()
            if estimated_size < 100:
                return (';', result.explode().unique().tolist())
    except:
        pass
    try:
        result = pd.Series(s).str.split(',')
        avg_words = result.str.len().mean()
        if avg_words > 1:
            estimated_size = result.explode().nunique()
            if estimated_size < 100:
                return (',', result.explode().unique().tolist())
    except:
        pass
    
    return False
         
class TypeInferFeatureGenerator():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx, df):
        ret_pa_fields = []
        pa_schema = pipeline[children[0]].output
        for pa_field, feature_name in tqdm(zip(pa_schema, df.columns), total=len(df.columns)):
            config = {}
            if try_datetime(df[feature_name]):
                config['is_datetime'] = True
            if try_category(df[feature_name]):
                config['is_categorical'] = True
            if try_numeric(df[feature_name]):
                config['is_numeric'] = True
            elif try_re_numeric(df[feature_name]):
                config['is_re_numeric'] = True
            config['is_list_string'] = try_list_string(df[feature_name])
            if config['is_list_string'] is False:
                skip = ('is_numeric' in config and config['is_numeric']) or ('is_re_numeric' in config and config['is_re_numeric'])
                if not skip:
                    config['is_onehot'] = try_onehot(df[feature_name])
            
            pa_field.copy_config_from(config)
            ret_pa_fields.append(pa_field)
        
        # append to pipeline
        cur_idx = max_idx + 1
        config = [x.mydump() for x in ret_pa_fields]
        pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_fields, op = 'type_infer', config = config)
        return pipeline, cur_idx, cur_idx
    
