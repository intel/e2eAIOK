"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core.schema import SeriesSchema
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
                        grouped_features[group_id] = {"is_timebased": False, "list": []}
                    grouped_features[group_id]["list"].append(pa_field.name)
                    if pa_field.is_timeseries:
                        grouped_features[group_id]["is_timebased"] = True

        # now, we build all groups
        for group_id, group_parts_dict in grouped_features.items():
            group_parts = group_parts_dict["list"]
            is_timebased = group_parts_dict["is_timebased"]
            out_schema = SeriesSchema(group_id, int)
            config = {}
            config['is_categorical'] = True
            config['is_timebased_categorical'] = is_timebased
            out_schema.copy_config_from(config)
            feature_in_out[group_id] = (group_parts, f"{folder}/{group_id}_categorify_dict.parquet", out_schema.name)
            is_useful = True
            ret_pa_schema.append(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            config = feature_in_out
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = 'group_categorify', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx