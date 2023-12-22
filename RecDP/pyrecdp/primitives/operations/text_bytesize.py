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

from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

def text_bytesize(s):
    return len(s.encode('utf-8'))

class TextBytesize(BaseLLMOperation):
    def __init__(self, text_key = 'text'):
        settings = {'text_key': text_key}
        requirements = []
        super().__init__(settings, requirements)
        self.text_key = text_key
        self.inplace = False
        self.support_spark = True
        self.support_ray = True        
        
    def process_rayds(self, ds):
        if self.inplace:
            raise NotImplementedError("We don't inplace modify text with normalization")
        else:
            new_name = 'bytesize'
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, text_bytesize))
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        bytesize_udf = F.udf(lambda x: len(x.encode('utf-8')), T.IntegerType())
        return spark_df.withColumn("bytesize", bytesize_udf(F.col(self.text_key)))
    
LLMOPERATORS.register(TextBytesize)