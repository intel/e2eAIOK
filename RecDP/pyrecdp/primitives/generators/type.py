from .base import BaseFeatureGenerator as super_class
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import pandas as pd
import numpy as np
from pandas.api import types as pdt
import copy

def try_category(s):
    if pdt.is_categorical_dtype(s) and not pdt.is_bool_dtype(s):
        return False
    n_unique = s.nunique()
    total_len = len(s)
    threshold = (total_len / 5) if (total_len / 5) < 10000 else 10000
    if 1 <= n_unique <= threshold:
        return True
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
        
class TypeInferFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx, df):
        ret_pa_fields = []
        pa_schema = pipeline[children[0]].output
        for pa_field, feature_name in zip(pa_schema, df.columns):
            config = {}
            if try_category(df[feature_name]):
                config['is_categorical'] = True
            if try_datetime(df[feature_name]):
                config['is_datetime'] = True
            pa_field.copy_config_from(config)
            ret_pa_fields.append(pa_field)
        
        # append to pipeline
        cur_idx = max_idx + 1
        config = [x.dump() for x in ret_pa_fields]
        pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_fields, op = 'type_infer', config = config)
        return pipeline, cur_idx, cur_idx
    

 
class TypeCheckFeatureGenerator(super_class):
    def __init__(self, final = False, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        config = {}
        for idx in range(len(pa_schema)):
            pa_field = pa_schema[idx]
            if pa_field.is_categorical_and_string:
                pa_schema[idx] = SeriesSchema(pa_field.name, pd.StringDtype())
                config[pa_field.name] = 'str'
            elif pa_field.is_categorical:
                pa_schema[idx] = SeriesSchema(pa_field.name, pd.Int32Dtype())
                config[pa_field.name] = 'int'
        
        cur_idx = max_idx
        #pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'type_check', config = config)
        return pipeline, cur_idx, cur_idx
