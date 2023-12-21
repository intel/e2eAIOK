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

class RenameOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.renamed = self.op.config
        self.support_spark_dataframe = True
        self.support_spark_rdd = True
        self.fast_without_dpp = True
        
    def get_function_pd(self, trans_type = 'fit_transform'):
        renamed = copy.deepcopy(self.renamed)
        def rename(df):
            return df.rename(columns = renamed)
        return rename

    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        def rename(df):
            for src, dst in self.renamed.items():
                df = df.withColumnRenamed(src, dst)
            
            return df
        return rename
AUTOFEOPERATORS.register(RenameOperation, "rename")