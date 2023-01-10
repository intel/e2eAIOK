from .base import BaseFeatureGenerator as super_class
from pyrecdp.core import SeriesSchema
import pandas as pd
import pyarrow as pa
from pandas.api import types as pdt
import numpy as np
from collections import OrderedDict
import inspect

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
   
    def is_useful(self, pa_schema):
        return True
    
    def fit_prepare(self, pa_schema, df):
        self._astype_feature_map = OrderedDict()
        ret_pa_fields = []
        for feature_name in df.columns:
            ret_field, type_change = self._infer_type(df[feature_name])
            if type_change:
                self.feature_in.append(feature_name)
                self._astype_feature_map[feature_name] = ret_field
            ret_pa_fields.append(ret_field)
        return ret_pa_fields

    def get_function_pd(self):
        def type_infer(df):            
            for feature_name in self.feature_in:
                df[feature_name] = convert_to_type(df[feature_name], self._astype_feature_map[feature_name])
            return df
        return type_infer

    def fit_transform(self, df):
        if not self._astype_feature_map:
            self.fit_prepare(None, df)
        return self.get_function_pd()(df)
 
    def dump_codes(self):
        codes = f"self._astype_feature_map = {self._astype_feature_map}\n"
        codes += inspect.getsource(self.get_function_pd())
        return codes
    
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