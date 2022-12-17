from .base import BaseFeatureGenerator as super_class
import pandas as pd
import pyarrow as pa
from pandas.api import types as pdt
import numpy as np
from collections import OrderedDict
import inspect

def convert_to_type(series, at):
    import pyarrow.types as types
    if types.is_timestamp(at):
        return pd.to_datetime(series, errors='coerce')
    elif types.is_dictionary(at):
        #TODO: this is not working with spark, need fix
        print("convert to categorical")
        return pd.Categorical(series)
    elif types.is_list(at):
        return pd.Series(series).str.split()
    return series

class TypeInferFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._astype_feature_map = None
   
    def is_useful(self, pa_schema):
        return True
    
    def fit_prepare(self, pa_schema, df):
        self._astype_feature_map = OrderedDict()
        ret_pa_fields = []
        for feature_name in df.columns:
            ret_field = self._infer_type(df[feature_name])
            self._astype_feature_map[feature_name] = ret_field
            ret_pa_fields.append(ret_field)
        return pa.schema(ret_pa_fields)

    def get_function_pd(self):
        def type_infer(df):            
            ret_list = OrderedDict()
            pa_schema = pa.Schema.from_pandas(df)
            for idx, feature_name in enumerate(df.columns):
                feature = df[feature_name]
                if pa_schema[idx] != self._astype_feature_map[feature_name]:
                    # do convert
                    feature = convert_to_type(feature, self._astype_feature_map[feature_name].type)
                ret_list[feature_name] = feature
            return pd.DataFrame(data = ret_list)
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
                return s
            n_unique = s.nunique()
            threshold = (n_unique / 5) if (n_unique / 5) < 1000 else 1000
            if 1 <= n_unique <= threshold:
                s = pd.Categorical(s)
            return s
            
        def try_datetime(s):
            if pdt.is_datetime64_any_dtype(s):
                return s
            if not pdt.is_string_dtype(s):
                return s
            if s.isnull().all():
                return s
            try:
                if len(s) > 500:
                    # Sample to speed-up type inference
                    result = s.sample(n=500, random_state=0)
                result = pd.to_datetime(result, errors='coerce')
                if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
                    return s
                else:
                    return result
            except:
                return s

        def try_text(s):
            if not pdt.is_string_dtype(s):
                return s
            if len(s) > 500:
                # Sample to speed-up type inference
                result = s.sample(n=500, random_state=0)
            try:
                avg_words = pd.Series(result).str.split().str.len().mean()
                if avg_words > 1:
                    # possible to use nlp method
                    s = pd.Series(result).str.split()
            except:
                return s
            return s
        
        s = try_category(s)
        s = try_datetime(s)
        s = try_text(s)
        return pa.field(s.name, pa.Array.from_pandas(s).type)
