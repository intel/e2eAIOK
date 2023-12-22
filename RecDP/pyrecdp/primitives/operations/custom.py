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

class CustomOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.custom_op = self.op.config["func_name"]
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self, trans_type = 'fit_transform'):
        return self.custom_op
AUTOFEOPERATORS.register(CustomOperation, "custom_operator")