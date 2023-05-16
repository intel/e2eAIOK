from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from featuretools.primitives import (
    Day,
    Month,
    Weekday,
    Year,
    Hour,
    PartOfDay
)
import numpy as np
import pandas as pd
from featuretools.primitives.base import TransformPrimitive
from pyrecdp.primitives.operations import Operation

class DatetimeTransformer(TransformPrimitive):
    name = "astype_datetime"
    return_type = np.datetime64

    def get_function(self):
        def convert(array):
            return pd.to_datetime(array, errors='coerce')

        return convert

class DatetimeFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_list = [
            Day,
            Month,
            Weekday,
            Year,
            Hour,
            #PartOfDay()
        ]
        self.op_name = 'datetime_feature'            

    def fit_prepare(self, pipeline, children, max_idx):
        cur_idx = max_idx
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if pa_field.is_datetime:
                self.feature_in.append(pa_field.name)
        feature_in_out_map = {}     
        for in_feat_name in self.feature_in:
            feature_in_out_map[in_feat_name] = []
            feature_in_out_map[in_feat_name].append((in_feat_name, DatetimeTransformer))
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = "astype", config = feature_in_out_map)
        return super().fit_prepare(pipeline, [cur_idx], cur_idx)
        