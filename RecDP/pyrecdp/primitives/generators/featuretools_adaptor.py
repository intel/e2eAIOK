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
from pyrecdp.core.schema import SeriesSchema
from pyrecdp.primitives.operations import Operation
import copy

class FeaturetoolsBasedFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []
        self.feature_in_out_map = {}
        self.op_name = 'featuretools'
    
    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        ret_pa_schema = copy.deepcopy(pa_schema)
        for in_feat_name in self.feature_in:
            is_useful = True
            self.feature_in_out_map[in_feat_name] = []
            for op in self.op_list:
                op_clz = op
                op = op_clz()
                out_feat_name = f"{in_feat_name}__{op.name}"
                out_feat_type = op.return_type
                out_schema = SeriesSchema(out_feat_name, out_feat_type)
                ret_pa_schema.append(out_schema)
                self.feature_in_out_map[in_feat_name].append((out_schema.name, op_clz))
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_schema, op = self.op_name, config = self.feature_in_out_map)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx