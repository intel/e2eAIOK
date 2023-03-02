from .base import BaseOperation
from pyrecdp.core import SeriesSchema
import pandas as pd

def convert_to_type(series, expected_schema: SeriesSchema):
    if expected_schema.is_datetime:
        return pd.to_datetime(series, errors='coerce')
    elif expected_schema.is_categorical:
        #TODO: this is not working with spark, need fix
        return pd.Categorical(series)
    return series

class TypeInferOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self._astype_feature_map = op_base.config

    def get_function_pd(self):
        def type_infer(df):            
            for feature_name, dest_type in self._astype_feature_map.items():
                df[feature_name] = convert_to_type(df[feature_name], dest_type)
            return df
        return type_infer