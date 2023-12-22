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
from .dataframe import *
from pyspark import RDD as SparkRDD

class MergeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config
        
    def execute_pd(self, pipeline, trans_type = 'fit_transform'):
        if self.cache is not None:
            return
        
        if len(self.op.children) != 2:
            raise ValueError("merge operation only accept num_children as 2")
        left_child = pipeline[self.op.children[0]].cache
        right_child = pipeline[self.op.children[1]].cache
        if isinstance(left_child, type(None)):
            print(f"left child is None, details: {pipeline[self.op.children[0]].describe()}")
        if isinstance(right_child, type(None)):
            print(f"right child is None, details: {pipeline[self.op.children[1]].describe()}")
        self.cache = pd.merge(left_child, right_child, on = self.config['on'], how = self.config['how'])

    def execute_spark(self, pipeline, rdp, trans_type = 'fit_transform'):
        if self.cache is not None:
            return
        
        if len(self.op.children) != 2:
            raise ValueError("merge operation only accept num_children as 2")
        
        self.convert(pipeline[self.op.children[0]], rdp)
        self.convert(pipeline[self.op.children[1]], rdp)
        left_child = pipeline[self.op.children[0]].cache
        right_child = pipeline[self.op.children[1]].cache
    
        self.cache = left_child.join(right_child, on = self.config['on'], how = self.config['how'])

    def convert(self, op, rdp):
        _convert = None
        df = op.cache
        if isinstance(df, SparkRDD):
            _convert = RDDToSparkDataFrameConverter().get_function(rdp)
        elif isinstance(df, pd.DataFrame):
            _convert = DataFrameToSparkDataFrameConverter().get_function(rdp)

        if _convert:
            df = _convert(df)
            op.cache = df

AUTOFEOPERATORS.register(MergeOperation, "merge")