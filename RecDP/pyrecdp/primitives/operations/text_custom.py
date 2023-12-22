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

from functools import partial
from typing import Optional, Dict

from .base import BaseLLMOperation, LLMOPERATORS, statistics_decorator
from ray.data import Dataset
from pyspark.sql import DataFrame
from .filter import BaseFilter

def text_bytesize(s):
    return len(s.encode('utf-8'))


class TextCustomerMap(BaseLLMOperation):
    def __init__(self, func, text_key='text', inplace: bool = False):
        settings = {'func': func, 'text_key': text_key, 'inplace': inplace}
        requirements = []
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        if not callable(func):
            import os
            if not os.path.exists(func):
                raise FileNotFoundError(f'Reload {func} object but not exists')
            import pickle
            with open(func, 'rb') as f:
                self.func = pickle.load(f)
        else:
            self.func = func
        self.text_key = text_key
        self.new_key = text_key if inplace else f"{self.func.__name__}_text"

    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds.map(lambda x: self.process_row(x, self.text_key, self.new_key, self.func))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        custom_udf = F.udf(self.func)
        return spark_df.withColumn(self.new_key, custom_udf(F.col(self.text_key)))


LLMOPERATORS.register(TextCustomerMap)


class TextCustomerFlatMap(BaseLLMOperation):
    def __init__(self, func, text_key='text', inplace: bool = False, **func_args):
        settings = {'func': func, 'text_key': text_key, 'inplace': inplace}
        settings.update(**func_args)
        requirements = []
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        if not callable(func):
            import os
            if not os.path.exists(func):
                raise FileNotFoundError(f'Reload {func} object but not exists')
            import pickle
            with open(func, 'rb') as f:
                self.func = pickle.load(f)
        else:
            self.func = func
        self.func = func
        self.text_key = text_key
        self.func_args = func_args
        self.new_key = text_key if inplace else f"{self.func.__name__}_text"

    def process_rayds(self, ds: Dataset) -> Dataset:
        def flap_map(sample, text_key, flat_map_func, func_args):
            result = []
            texts = flat_map_func(sample[text_key], **func_args) if func_args else flat_map_func(sample[text_key])
            for text in texts:
                row = dict(**sample)
                row[self.new_key] = text
                result.append(row)
            return result

        return ds.flat_map(lambda sample: flap_map(sample, self.text_key, self.func, self.func_args))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T

        flat_map_udf = F.udf(self.func, T.ArrayType(T.StringType()))
        spark_df = spark_df.withColumn(self.new_key, flat_map_udf(F.col(self.new_key)))
        spark_df = spark_df.withColumn(self.new_key, F.explode(F.col(self.new_key)))
        return spark_df


LLMOPERATORS.register(TextCustomerFlatMap)


class TextCustomerFilter(BaseFilter):
    def __init__(self, func, text_key='text'):
        settings = {'func': func, 'text_key': text_key}
        requirements = []
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        if not callable(func):
            import os
            if not os.path.exists(func):
                raise FileNotFoundError(f'Reload {func} object but not exists')
            import pickle
            with open(func, 'rb') as f:
                self.func = pickle.load(f)
        else:
            self.func = func
        self.text_key = text_key

    @statistics_decorator
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            # remove unwanted text row inplace
            filtered_ds = ds.filter(lambda x: self.func(x[self.text_key]))
            return filtered_ds
        else:
            raise NotImplementedError(f"We only support inplace modification for {self.__class__.__name__}.")

    @statistics_decorator
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        if self.inplace:
            compute_udf = F.udf(self.func, T.BooleanType())
            return spark_df.filter(compute_udf(F.col(self.text_key)))
        else:
            raise NotImplementedError(f"We only support inplace modification for {self.__class__.__name__}.")


LLMOPERATORS.register(TextCustomerFilter)
