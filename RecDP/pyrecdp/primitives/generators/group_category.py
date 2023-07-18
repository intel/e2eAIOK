from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import copy

class GroupCategoryFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        grouped_features = {}
        feature_in_out = {}
        folder = 'pipeline_default'
        ret_pa_schema = copy.deepcopy(pa_schema)
        for pa_field in pa_schema:
            if pa_field.is_grouped_categorical:                
                for group_id in pa_field.group_id_list:
                    if group_id not in grouped_features:
                        grouped_features[group_id] = []
                    grouped_features[group_id].append(pa_field.name)
        # now, we build all groups
        for group_id, group_parts in grouped_features.items():
            out_schema = SeriesSchema(group_id, int)
            config = {}
            config['is_categorical'] = True
            out_schema.copy_config_from(config)
            feature_in_out[group_id] = (group_parts, f"{folder}/{group_id}_categorify_dict", out_schema.name)
            is_useful = True
            ret_pa_schema.append(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            config = feature_in_out
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'group_categorify', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx