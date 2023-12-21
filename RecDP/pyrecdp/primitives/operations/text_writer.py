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
import os
import shutil

class ParquetWriter(BaseLLMOperation):
    def __init__(self, output_dir):
        settings = {'output_dir': output_dir}
        requirements = []
        super().__init__(settings, requirements)
        self.support_ray = True
        self.support_spark = True
        self.output_dir = output_dir
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if os.path.exists(self.output_dir) and os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        ds.write_parquet(self.output_dir)
        return ds
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        spark_df.write.parquet(self.output_dir, mode='overwrite')
        return spark_df
    
LLMOPERATORS.register(ParquetWriter)

class JsonlWriter(BaseLLMOperation):
    def __init__(self, output_dir):
        settings = {'output_dir': output_dir}
        requirements = []
        super().__init__(settings, requirements)
        self.support_ray = True
        self.support_spark = True
        self.output_dir = os.path.join(output_dir, 'output')

    def process_rayds(self, ds: Dataset) -> Dataset:
        if os.path.exists(self.output_dir) and os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        ds.write_json(self.output_dir)
        return ds

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        spark_df.write.json(self.output_dir, mode='overwrite')
        return spark_df

LLMOPERATORS.register(JsonlWriter)

class ClassifyParquetWriter(BaseLLMOperation):
    def __init__(self, output_dir, key):
        settings = {'output_dir': output_dir, 'key': key}
        requirements = []
        super().__init__(settings, requirements)
        self.support_ray = False
        self.support_spark = True
        self.output_dir = output_dir
        self.key = key
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        spark_df.write.mode("overwrite").partitionBy(self.key).parquet(self.output_dir)
        return spark_df
    
LLMOPERATORS.register(ClassifyParquetWriter)

class ClassifyJsonlWriter(BaseLLMOperation):
    def __init__(self, output_dir, key):
        settings = {'output_dir': output_dir, 'key': key}
        requirements = []
        super().__init__(settings, requirements)
        self.support_ray = False
        self.support_spark = True
        self.output_dir = output_dir
        self.key = key

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        spark_df.write.mode("overwrite").partitionBy(self.key).json(self.output_dir)
        return spark_df

LLMOPERATORS.register(ClassifyJsonlWriter)

class PerfileParquetWriter(BaseLLMOperation):
    def __init__(self, output_dir):
        settings = {'output_dir': output_dir}
        requirements = []
        super().__init__(settings, requirements)
        self.support_ray = True
        self.support_spark = True
        self.output_dir = output_dir
        
    def execute_ray(self, pipeline, source_id):
        child_output = []
        children = self.op.children if self.op.children is not None else []
        for op in children:
            child_output.append(pipeline[op].cache)
        self.cache = self.process_rayds(source_id, *child_output)
        return self.cache

    def process_rayds(self, source_id, ds: Dataset) -> Dataset:
        to_save = os.path.join(self.output_dir, source_id)
        if os.path.exists(to_save) and os.path.isdir(to_save):
            shutil.rmtree(to_save)
        ds.write_parquet(to_save)
        return ds

    def execute_spark(self, pipeline, source_id):
        child_output = []
        children = self.op.children if self.op.children is not None else []
        for op in children:
            child_output.append(pipeline[op].cache)
        self.cache = self.process_spark(source_id, *child_output)
        return self.cache

    def process_spark(self, source_id, spark_df: DataFrame = None) -> DataFrame:
        to_save = os.path.join(self.output_dir, source_id)
        spark_df.write.mode("overwrite").parquet(to_save)
        return spark_df

LLMOPERATORS.register(PerfileParquetWriter)

class PerfileJsonlWriter(BaseLLMOperation):
    def __init__(self, output_dir):
        settings = {'output_dir': output_dir}
        requirements = []
        super().__init__(settings, requirements)
        self.support_ray = True
        self.support_spark = True
        self.output_dir = output_dir
        
    def execute_ray(self, pipeline, source_id):
        child_output = []
        children = self.op.children if self.op.children is not None else []
        for op in children:
            child_output.append(pipeline[op].cache)
        self.cache = self.process_rayds(source_id, *child_output)
        return self.cache
        
    def process_rayds(self, source_id, ds: Dataset) -> Dataset:
        to_save = os.path.join(self.output_dir, source_id)
        if os.path.exists(to_save) and os.path.isdir(to_save):
            shutil.rmtree(to_save)
        ds.write_json(to_save)
        return ds
    
    def execute_spark(self, pipeline, source_id):
        child_output = []
        children = self.op.children if self.op.children is not None else []
        for op in children:
            child_output.append(pipeline[op].cache)
        self.cache = self.process_spark(source_id, *child_output)
        return self.cache

    def process_spark(self, source_id, spark_df: DataFrame = None) -> DataFrame:
        to_save = os.path.join(self.output_dir, source_id)
        spark_df.write.mode("overwrite").json(to_save)
        return spark_df

LLMOPERATORS.register(PerfileJsonlWriter)