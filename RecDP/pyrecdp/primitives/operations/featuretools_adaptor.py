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

from .base import BaseOperation
import copy
from pyrecdp.core.utils import class_name_fix

class FeaturetoolsOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out_map = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in_out_map = copy.deepcopy(self.feature_in_out_map)
        def generate_ft_feature(df):
            for in_feat_name, ops in feature_in_out_map.items():
                if in_feat_name in df.columns:
                    for op in ops:
                        op_object = class_name_fix(op[1])()
                        df[op[0]] = op_object(df[in_feat_name])
            return df
        return generate_ft_feature
    
    def get_function_spark(self, rdp):
        raise NotImplementedError(f"operations based on featuretools are not support Spark DataFrame yet.")