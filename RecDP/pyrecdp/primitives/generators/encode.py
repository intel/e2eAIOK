from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation

class LabelEncodeFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        feature_in_out = {}
        folder = 'pipeline_default'
        for idx, pa_field in enumerate(pa_schema):
            if pa_field.is_string:
                feature = pa_field.name
                out_schema = SeriesSchema(f"{feature}", pd.CategoricalDtype())
                feature_in_out[feature] = (f"{folder}/{feature}_categorify_dict", feature)
                is_useful = True
                pa_schema[idx] = out_schema
        if is_useful:
            cur_idx = max_idx + 1
            config = feature_in_out
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'categorify', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx

class OneHotFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        config = {}
        for pa_field in pa_schema:
            if pa_field.is_onehot:
                feature = pa_field.name
                out_schema = [SeriesSchema(f"{feature}__{key}", int) for key in pa_field.config["is_onehot"]]
                config[pa_field.name] = pa_field.config["is_onehot"]
                is_useful = True
                pa_schema.extend(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'onehot_encode', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx
        
class ListOneHotFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        config = {}
        for pa_field in pa_schema:
            if pa_field.is_list_string:
                feature = pa_field.name
                out_schema = [SeriesSchema(f"{feature}_{key}", int) for key in pa_field.config["is_list_string"][1] if key != None or key != ""]
                config[pa_field.name] = pa_field.config["is_list_string"]
                is_useful = True
                pa_schema.extend(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'list_onehot_encode', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx
        
class TargetEncodeFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        config = {}
        for pa_field in pa_schema:
            if pa_field.is_onehot:
                feature = pa_field.name
                out_schema = [SeriesSchema(f"{feature}__{key}", int) for key in pa_field.config["is_onehot"]]
                config[pa_field.name] = pa_field.config["is_onehot"]
                is_useful = True
                pa_schema.extend(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'categorify', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx