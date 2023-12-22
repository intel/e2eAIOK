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

from pyspark.sql import DataFrame
from ray.data import Dataset

from pyrecdp.primitives.operations.base import BaseLLMOperation, statistics_decorator
import pyspark.sql.functions as F


class BaseFilter(BaseLLMOperation):
    def __init__(self, args_dict={}, requirements=[]):
        super().__init__(args_dict=args_dict, requirements=requirements)
        self.text_key = 'text'
        self.inplace = True
        self.support_ray = True
        self.support_spark = True

    @statistics_decorator
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            # remove unwanted text row inplace
            compute_func = self.get_compute_func()
            filtered_ds = ds.filter(lambda x: compute_func(x[self.text_key]))
            return filtered_ds
        else:
            raise NotImplementedError(f"We only support inplace modification for {self.__class__.__name__}.")

    @statistics_decorator
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        if self.inplace:
            import pyspark.sql.types as T
            compute_udf = F.udf(self.get_compute_func(), T.BooleanType())
            return spark_df.filter(compute_udf(F.col(self.text_key)))
        else:
            raise NotImplementedError(f"We only support inplace modification for {self.__class__.__name__}.")

    def get_compute_func(self, *args, **kwargs):
        raise NotImplementedError("Abstract func")
