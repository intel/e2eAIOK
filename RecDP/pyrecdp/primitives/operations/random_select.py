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

class RandomSelect(BaseLLMOperation):
    def __init__(self, fraction=1.0,seed=42):
        settings = {'fraction': fraction}
        super().__init__(settings)
        self.fraction=fraction
        self.seed=seed
        self.support_spark = True
        self.support_ray = True
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds.random_sample(fraction=self.fraction, seed=self.seed)
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        return spark_df.sample(fraction=self.fraction, seed=self.seed)
    
LLMOPERATORS.register(RandomSelect)