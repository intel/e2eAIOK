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
from pyrecdp.core.schema import SeriesSchema
import pandas as pd
import copy

class TypeInferOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.astype_feature_map = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True

    def get_function_pd(self, trans_type = 'fit_transform'):
        astype_feature_map = copy.deepcopy(self.astype_feature_map)
        def type_infer(df):
            for feature in astype_feature_map:
                feature_name, dest_type_list = feature[0], feature[1]
                # if 'is_datetime' in dest_type_list:
                #     df[feature_name] = pd.to_datetime(df[feature_name], errors='coerce', infer_datetime_format=True)
                if 'is_numeric' in dest_type_list:
                    df[feature_name] = pd.to_numeric(df[feature_name], errors='coerce')
            return df
        return type_infer
    
    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        raise NotImplementedError(f"TypeInferOperation spark dataframe is not supported yet.")
AUTOFEOPERATORS.register(TypeInferOperation, "type_infer")