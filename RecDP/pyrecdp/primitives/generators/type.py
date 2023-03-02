from .base import BaseFeatureGenerator as super_class
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import pandas as pd
from pandas.api import types as pdt

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD

def convert_to_type(series, expected_schema: SeriesSchema):
    if expected_schema.is_datetime:
        return pd.to_datetime(series, errors='coerce')
    elif expected_schema.is_categorical:
        #TODO: this is not working with spark, need fix
        return pd.Categorical(series)
    return series

class TypeInferFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._astype_feature_map = None
        self.feature_in = []
   
    def fit_prepare(self, pipeline, children, max_idx, df):
        self._astype_feature_map = {}
        ret_pa_fields = []
        for feature_name in df.columns:
            ret_field, type_change = self._infer_type(df[feature_name])
            if type_change:
                self.feature_in.append(feature_name)
                self._astype_feature_map[feature_name] = ret_field
            ret_pa_fields.append(ret_field)
        
        # append to pipeline
        cur_idx = max_idx + 1
        config = self._astype_feature_map
        pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_fields, op = 'type_infer', config = config)
        return pipeline, cur_idx, cur_idx
    
    def _infer_type(self, s):
        def try_category(s):
            if pdt.is_categorical_dtype(s) and not pdt.is_bool_dtype(s):
                return s, False
            n_unique = s.nunique()
            total_len = len(s)
            threshold = (total_len / 5) if (total_len / 5) < 10000 else 10000
            if 1 <= n_unique <= threshold:
                return s.astype("category"), True
            return s, False
            
        def try_datetime(s):
            if pdt.is_datetime64_any_dtype(s):
                return s, False
            if not pdt.is_string_dtype(s):
                return s, False
            if s.isnull().all():
                return s, False
            try:
                if len(s) > 500:
                    # Sample to speed-up type inference
                    result = s.sample(n=500, random_state=0)
                result = pd.to_datetime(result, errors='coerce')
                if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
                    return s, False
                else:
                    return result, True
            except:
                return s, False
        
        type_change = False
        if not type_change:
            s, type_change = try_category(s)
        if not type_change:
            s, type_change = try_datetime(s)    
            
        return SeriesSchema(s), type_change
 
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
