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
from pyrecdp.core.schema import SeriesSchema

class DatetimeTransformer(TransformPrimitive):
    name = "astype_datetime"
    return_type = np.datetime64

    def get_function(self):
        def convert(array):
            return pd.to_datetime(array, errors='coerce', infer_datetime_format=True)

        return convert

class DatetimeFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_list = {
            'day': Day,
            'month': Month,
            'weekday': Weekday,
            'year': Year,
            'hour': Hour,
            #PartOfDay()
        }
        self.op_name = 'datetime_feature'

    def fit_prepare(self, pipeline, children, max_idx):
        cur_idx = max_idx
        pa_schema = pipeline[children[0]].output
        feature_in = {}
        for pa_field in pa_schema:
            if pa_field.is_datetime:                
                feature_in[pa_field.name] = [self.op_list[i] for i in pa_field.datetime_ft_list]
            
        feature_in_out_map = {}
        feature_in_for_ops = {}
        for in_feat_name in feature_in:
            out_feat_name = f"{in_feat_name}_dt"
            feature_in_out_map[in_feat_name] = []
            feature_in_out_map[in_feat_name].append((out_feat_name, DatetimeTransformer))
            out_schema = SeriesSchema(out_feat_name, np.datetime64)
            pa_schema.append(out_schema)
            cur_idx = cur_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = "astype", config = feature_in_out_map)
            children = [cur_idx]
            feature_in_for_ops[out_feat_name] = feature_in[in_feat_name]
            
        is_useful = False
        for in_feat_name in feature_in_for_ops:
            is_useful = True
            self.feature_in_out_map[in_feat_name] = []
            for op in feature_in_for_ops[in_feat_name]:
                op_clz = op
                op = op_clz()
                out_feat_name = f"{in_feat_name}__{op.name}"
                out_feat_type = op.return_type
                out_schema = SeriesSchema(out_feat_name, out_feat_type)
                pa_schema.append(out_schema)
                self.feature_in_out_map[in_feat_name].append((out_schema.name, op_clz))
        if is_useful:
            cur_idx = cur_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = self.op_name, config = self.feature_in_out_map)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx
        