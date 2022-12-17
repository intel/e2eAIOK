from .base import BaseFeatureGenerator as super_class
import pandas as pd
import inspect

def get_default_value(at):
    import pyarrow.types as types
    if types.is_boolean(at):
        return False
    elif types.is_int8(at):
        return -1
    elif types.is_int16(at):
        return -1
    elif types.is_int32(at):
        return -1
    elif types.is_int64(at):
        return -1
    elif types.is_float32(at):
        return -1
    elif types.is_float64(at):
        return -1
    elif types.is_string(at):
        return ""
    elif types.is_date32(at):
        return pd.Timestamp(0)
    elif types.is_timestamp(at):
        return pd.Timestamp(0)
    return None
    

class FillNaFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return True
    
    def fit_prepare(self, pa_schema):
        self._fillna_feature_map = {}
        for field in pa_schema:
            default_value = get_default_value(field.type)
            if default_value:
                self._fillna_feature_map[field.name] = default_value
        return pa_schema
    
    def get_function_pd(self):
        def fill_na(df):
            df.fillna(self._fillna_feature_map, inplace=True, downcast=False)
            return df
        return fill_na
 
    def dump_codes(self):
        codes = f"self._fillna_feature_map = {self._fillna_feature_map}\n"
        codes += inspect.getsource(self.get_function_pd())
        return codes

