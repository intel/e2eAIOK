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
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD
 
class DropUselessFeatureGenerator(super_class):
    def __init__(self, final = False, **kwargs):
        super().__init__(**kwargs)
        self._astype_feature_map = None
        self.feature_in = []
        self.final = final

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if not self.final:
                if not (pa_field.is_numeric or pa_field.is_categorical):
                    self.feature_in.append(pa_field.name)
                    is_useful = True
            else:
                if not (pa_field.is_numeric):
                    self.feature_in.append(pa_field.name)
                    is_useful = True
        ret_schema = []
        for pa_field in pa_schema:
            if pa_field.name not in self.feature_in:
                ret_schema.append(pa_field)
        if is_useful:
            cur_idx = max_idx + 1
            config = self.feature_in
            pipeline[cur_idx] = Operation(cur_idx, children, ret_schema, op = 'drop', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx