from .base import BaseFeatureGenerator as super_class
from pyrecdp.primitives.utils import SeriesSchema
from pyrecdp.primitives import utils
import pyarrow.types as types
from featuretools.primitives import (
    Day,
    Month,
    Weekday,
    Year,
    Hour,
    PartOfDay
)

class DatetimeFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []
        self.feature_in_out_map = {}
        self.op_list = [
            Day(),
            Month(),
            Weekday(),
            Year(),
            Hour(),
            PartOfDay()
        ]

    def is_useful(self, pa_schema):
        found = False
        for pa_field in pa_schema:
            if pa_field.is_datetime:
                self.feature_in.append(pa_field.name)
                found = True
        return found
    
    def fit_prepare(self, pa_schema):
        for in_feat_name in self.feature_in:
            self.feature_in_out_map[in_feat_name] = []
            for op in self.op_list:
                out_feat_name = f"{in_feat_name}.{op.name}"
                out_feat_type = op.return_type
                self.feature_in_out_map[in_feat_name].append((SeriesSchema(out_feat_name, out_feat_type), op))
        return pa_schema

    def get_function_pd(self):
        def generate_datetime_feature(df):
            for in_feat_name in self.feature_in:
                for op in self.feature_in_out_map[in_feat_name]:
                    df[op[0].name] = op[1](df[in_feat_name])
            return df
        return generate_datetime_feature