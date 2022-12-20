from .base import BaseFeatureGenerator as super_class
from pyrecdp.primitives.utils import SeriesSchema
import pandas as pd
import inspect

def get_default_value(at: SeriesSchema):
    if at.is_boolean:
        return False
    elif at.is_numeric:
        return -1
    elif at.is_string:
        return ""
    elif at.is_datetime:
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
            default_value = get_default_value(field)
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

