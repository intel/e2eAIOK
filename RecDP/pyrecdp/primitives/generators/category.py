from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import copy

class CategoryFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        feature_in_out = {}
        folder = 'pipeline_default'
        ret_pa_schema = copy.deepcopy(pa_schema)
        for pa_field in pa_schema:
            if pa_field.is_categorical_and_string:
                feature = pa_field.name
                out_schema = SeriesSchema(f"{feature}__idx", pd.CategoricalDtype())
                feature_in_out[feature] = (f"{folder}/{feature}_categorify_dict", out_schema.name)
                is_useful = True
                ret_pa_schema.append(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            config = feature_in_out
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'categorify', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx