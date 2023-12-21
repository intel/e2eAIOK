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

from .base import BaseOperation, AUTOFEOPERATORS
from .featuretools_adaptor import FeaturetoolsOperation
import copy
from pyrecdp.core.utils import class_name_fix

class HaversineOperation(FeaturetoolsOperation):
    def __init__(self, op_base):
        super().__init__(op_base)

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in_out_map = copy.deepcopy(self.feature_in_out_map)
        def generate_ft_feature(df):
            for inputs_str, op in feature_in_out_map.items():
                inputs = eval(inputs_str)
                op_object = class_name_fix(op[1])()
                df[op[0]] = op_object(df[inputs[0]], df[inputs[1]])
            return df
        return generate_ft_feature
AUTOFEOPERATORS.register(HaversineOperation, "haversine")