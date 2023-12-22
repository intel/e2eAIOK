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
import copy

class FillNaOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self._fillna_feature_map = self.op.config
        self.support_spark_dataframe = True
        self.support_spark_rdd = True
    
    def get_function_pd(self, trans_type = 'fit_transform'):
        _fillna_feature_map = copy.deepcopy(self._fillna_feature_map)
        def fill_na(df):
            df.fillna(_fillna_feature_map, inplace=True, downcast=False)
            return df
        return fill_na

    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        def fill_na(df):
            return df.na.fill(self._fillna_feature_map)
        return fill_na
AUTOFEOPERATORS.register(FillNaOperation, "fillna")