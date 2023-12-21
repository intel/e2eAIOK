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
from pyrecdp.primitives.operations import Operation
    
class RenameFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renamed = {}

    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        is_useful = False
        for pa_field in pa_schema:
            feature_name = pa_field.name
            if '.' in feature_name:
                feature_name = feature_name.replace('.', '__')
                is_useful = True
            if ' ' in feature_name:
                feature_name = feature_name.replace(' ', '_')
                is_useful = True
            if '?' in feature_name:
                feature_name = feature_name.replace('?', '')
                is_useful = True
            if ',' in feature_name:
                feature_name = feature_name.replace(',', '')
                is_useful = True
            if is_useful:
                self.renamed[pa_field.name] = feature_name
        ret_schema = []
        for pa_field in pa_schema:
            if pa_field.name in self.renamed:
                pa_field.name = self.renamed[pa_field.name]
            ret_schema.append(pa_field)

        if is_useful:
            cur_idx = max_idx + 1
            config = self.renamed
            pipeline[cur_idx] = Operation(cur_idx, children, ret_schema, op = 'rename', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx