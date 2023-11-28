from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core.schema import SeriesSchema
from pyrecdp.primitives.operations import Operation
import copy

class LabelEncodeFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        ret_pa_schema = copy.deepcopy(pa_schema)
        feature_in_out = {}
        folder = 'pipeline_default'
        for idx, pa_field in enumerate(pa_schema):
            if pa_field.is_string and pa_field.is_label:
                feature = pa_field.name
                config = {}
                config = pa_field.copy_config_to(config)
                config['is_string'] = False
                out_schema = SeriesSchema(f"{feature}", int)
                out_schema.copy_config_from(config)
                feature_in_out[feature] = (f"{folder}/{feature}_categorify_dict", feature)
                is_useful = True
                ret_pa_schema[idx] = out_schema
        if is_useful:
            cur_idx = max_idx + 1
            config = feature_in_out
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'categorify', config = config)
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
        ret_pa_schema = copy.deepcopy(pa_schema)
        config = {}
        for pa_field in pa_schema:
            if pa_field.is_onehot:
                feature = pa_field.name
                out_schema = [SeriesSchema(f"{feature}__{key}", int) for key in pa_field.config["is_onehot"]]
                config[pa_field.name] = pa_field.config["is_onehot"]
                is_useful = True
                ret_pa_schema.extend(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'onehot_encode', config = config)
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
        ret_pa_schema = copy.deepcopy(pa_schema)
        config = {}
        for pa_field in pa_schema:
            if pa_field.is_list_string:
                feature = pa_field.name
                out_schema = [SeriesSchema(f"{feature}_{key}", int) for key in pa_field.config["is_list_string"][1] if key != None or key != ""]
                config[pa_field.name] = pa_field.config["is_list_string"]
                is_useful = True
                ret_pa_schema.extend(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'list_onehot_encode', config = config)
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
        feature_in_out = {}
        ret_pa_schema = copy.deepcopy(pa_schema)
        folder = 'pipeline_default'
        for pa_field in pa_schema:
            if pa_field.is_categorical and not pa_field.is_timebased_categorical:
                feature = pa_field.name
                out_schema = SeriesSchema(f"{feature}__TE", float)
                feature_in_out[feature] = (f"{folder}/{feature}_TE_dict.pkl", out_schema.name)
                is_useful = True
                ret_pa_schema.append(out_schema)
        if is_useful:
            # find label column
            label_list = [pa_field.name for pa_field in pa_schema if pa_field.is_label]
            cur_idx = max_idx + 1
            config = {}
            config['feature_in_out'] = feature_in_out
            config['label'] = label_list[0]
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'target_encode', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx

class CountEncodeFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        feature_in_out = {}
        ret_pa_schema = copy.deepcopy(pa_schema)
        folder = 'pipeline_default'
        for pa_field in pa_schema:
            if pa_field.is_categorical:
                feature = pa_field.name
                out_schema = SeriesSchema(f"{feature}__CE", int)
                feature_in_out[feature] = (f"{folder}/{feature}_CE_dict.pkl", out_schema.name)
                is_useful = True
                ret_pa_schema.append(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            config = feature_in_out
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'count_encode', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx