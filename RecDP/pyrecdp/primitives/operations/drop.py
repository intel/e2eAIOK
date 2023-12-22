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
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD
import copy
from pyrecdp.core.utils import is_unique
 
class DropOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = self.op.config
        self.support_spark_dataframe = True
        self.support_spark_rdd = True
        self.fast_without_dpp = True

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in = copy.deepcopy(self.feature_in)

        def drop_useless_feature(df):
            return df.drop(columns = feature_in)
            # for i in df.columns:
            #     if i not in feature_in and is_unique(df[i]):
            #         feature_in.append(i)
            # return df.drop(columns = feature_in)
        return drop_useless_feature
    
    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        def drop_feature(df):
            return df.drop(*self.feature_in)
        return drop_feature
AUTOFEOPERATORS.register(DropOperation, "drop")