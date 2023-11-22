from pyrecdp.core.schema import SeriesSchema
from pyrecdp.core.utils import is_unique 
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
    if pdt.is_float_dtype(s):
        return False
    n_unique = s.nunique()
    total_len = len(s)
    threshold = (total_len / 5)
    if 1 < n_unique <= threshold:
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
        else:
            result = s
        result = pd.to_datetime(result, errors='coerce')
        if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
            return False
        else:
            return True
    except:
        return False
    
def get_datetime_potential_features(s):
    if len(s) > 5000:
        # Sample to speed-up type inference
        result = s.sample(n=5000, random_state=0)
    else:
        result = s
    result = pd.to_datetime(result, errors='coerce')
    ret = []
    if not is_unique(result.dt.day):
        ret.append('day')
    if not is_unique(result.dt.month):
        ret.append('month')
    if not is_unique(result.dt.year):
        ret.append('year')
    if not is_unique(result.dt.weekday):
        ret.append('weekday')
    if not is_unique(result.dt.hour):
        ret.append('hour')    
    return ret

def try_re_numeric(s):
    if s.isnull().all():
        return True
    try:
        if len(s) > 500:
            # Sample to speed-up type inference
            result = s.sample(n=500, random_state=0)
        else:
            result = s
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
        else:
            result = s
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
 
def is_encoded(s):
    if len(s) > 1000:
        sample_data = s.sample(n=1000, random_state=0)
    else:
        sample_data = s
    from pyrecdp.primitives.generators.nlp import BertTokenizerDecode
    proc_ = BertTokenizerDecode().get_function()
    try:
        proc_(sample_data)
    except Exception as e:
        #print(e)
        return False
    return True
         
class TypeInferFeatureGenerator():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx, df, y = None, ts = None):
        ret_pa_fields = []
        pa_schema = pipeline[children[0]].output
        for pa_field, feature_name in tqdm(zip(pa_schema, df.columns), total=len(df.columns)):
            config = {}
            # Add y_label to y column
            if feature_name == y:
                config['is_label'] = True
                if pa_field.is_string:
                    config['is_categorical_label'] = True
            else:
                if isinstance(ts, str) and feature_name == ts:
                    config['is_timeseries'] = True
                if isinstance(ts, list) and feature_name in ts:
                    config['is_timeseries'] = True
                # if pa_field.is_text and is_encoded(df[feature_name]):
                #     config['is_encoded'] = True
                if try_datetime(df[feature_name]):
                    config['is_datetime'] = True
                    config['datetime_ft'] = get_datetime_potential_features(df[feature_name])
                if try_category(df[feature_name]):
                    config['is_categorical'] = True
                if try_numeric(df[feature_name]):
                    config['is_numeric'] = True
                elif try_re_numeric(df[feature_name]):
                    config['is_re_numeric'] = True
                    config['is_categorical'] = False
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
    
