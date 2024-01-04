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
from loguru import logger

class DatasetReader(BaseLLMOperation):
    def __init__(self):
        super().__init__()
        self.support_ray = True
        self.support_spark = True

LLMOPERATORS.register(DatasetReader)

class TextReader(BaseLLMOperation):
    def __init__(self, args_dict={}, requirements=[]):
        super().__init__(args_dict, requirements)
        self.column_rename_dict = {}

    def rename_ray_ds_columns(self, ds):
        def add_column_with_new_name(content, rename_dict):
            for pre_name, new_name in rename_dict.items():
                content[new_name] = content[pre_name]
            return content
        pre_columns = [column for column in self.column_rename_dict.keys() if column in ds.columns()]
        rename_dict = {}
        for pre_column in pre_columns:
            rename_dict[pre_column] = self.column_rename_dict[pre_column]
        ds = ds.map(lambda x: add_column_with_new_name(x, rename_dict))
        ds = ds.drop_columns(pre_columns)
        return ds

    def rename_spark_df_columns(self, df):
        for pre_column, new_column in self.column_rename_dict.items():
            if pre_column in df.columns:
                df = df.withColumnRenamed(pre_column, new_column)
        return df

    def union_ray_ds(self, ds1, ds2):
        if (not isinstance(ds1, Dataset)) and (not isinstance(ds2, Dataset)):
            raise ValueError(f"union_spark_df both arguments are not DataFrame, df1 is {type(ds1)}, df2 is {type(ds2)}")
        if (not isinstance(ds1, Dataset)) or (not isinstance(ds1, Dataset)):
            return ds1 if isinstance(ds1, Dataset) else ds2
        def add_new_empty_column(content, column_name):
            content[column_name] = None
            return content
        def convert_to_string(content, column_name):
            content[column_name] = str(content[column_name])
            return content
        for column in [column for column in ds2.columns() if column not in ds1.columns()]:
            ds1 = ds1.map(lambda x: add_new_empty_column(x, column))
        for column in [column for column in ds1.columns() if column not in ds2.columns()]:
            ds2 = ds2.map(lambda x: add_new_empty_column(x, column))
        ds1_fields_dict  =dict(zip(ds1.schema().names, ds1.schema().types))
        ds2_fields_dict = dict(zip(ds2.schema().names, ds2.schema().types))
        for column_name in ds1_fields_dict.keys():
            if ds2_fields_dict[column_name] != ds1_fields_dict[column_name] and not (
                    str(ds2_fields_dict[column_name]) == "null" or str(ds1_fields_dict[column_name]) == "null"):
                ds1 = ds1.map(lambda x: convert_to_string(x, column_name))
                ds2 = ds2.map(lambda x: convert_to_string(x, column_name))
        return ds1.union(ds2)

    def union_spark_df(self, df1, df2):
        if (not isinstance(df1, DataFrame)) and (not isinstance(df1, DataFrame)):
            raise ValueError(f"union_spark_df both arguments are not DataFrame, df1 is {type(df1)}, df2 is {type(df2)}")
        if (not isinstance(df1, DataFrame)) or (not isinstance(df1, DataFrame)):
            return df1 if isinstance(df1, DataFrame) else df2
        from pyspark.sql.functions import lit
        import pyspark.sql.functions as F
        from pyspark.sql.types import NullType, StringType
        for column in [column for column in df2.columns if column not in df1.columns]:
            df1 = df1.withColumn(column, lit(None))
        for column in [column for column in df1.columns if column not in df2.columns]:
            df2 = df2.withColumn(column, lit(None))
        df1_fields, df2_fields = df1.schema.fields, df2.schema.fields
        df1_fields_dict, df2_fields_dict = {}, {}
        for df1_field in df1_fields:
            df1_fields_dict[df1_field.name] = df1_field.dataType
        for df2_field in df2_fields:
            df2_fields_dict[df2_field.name] = df2_field.dataType
        for column_name in df1_fields_dict.keys():
            if df2_fields_dict[column_name] != df1_fields_dict[column_name] and not (
                    df2_fields_dict[column_name] == NullType() or df1_fields_dict[column_name] == NullType()):
                df1 = df1.withColumn(column_name, F.col(column_name).cast(StringType()))
                df2 = df2.withColumn(column_name, F.col(column_name).cast(StringType()))
        return df1.unionByName(df2, allowMissingColumns=True)

class JsonlReader(TextReader):
    def __init__(self, input_dir = "", column_rename_dict = {}):
        settings = {'input_dir': input_dir, 'column_rename_dict': column_rename_dict}
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.input_dir = input_dir
        self.column_rename_dict = column_rename_dict

    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        self.cache = self.rename_ray_ds_columns(rd.read_json(self.input_dir))
        self.statistics.total_in = self.cache.count()
        self.statistics.total_out = self.statistics.total_in
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        df = self.rename_spark_df_columns(spark.read.json(self.input_dir).cache())
        self.statistics.total_in = df.count()
        if '_corrupt_record' in df.columns:
            df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out = df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
        else:
            self.statistics.total_out = self.statistics.total_in
        self.cache = df
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache

    def summarize(self) -> str:
        self.get_modified_rows()
        return (
            f"A total of {self.statistics.total_in} rows of json data were processed, "
            f"with {self.statistics.total_changed} rows read corrupted, {self.statistics.total_out} rows of data remaining.")

LLMOPERATORS.register(JsonlReader)

class SourcedReader(TextReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        settings = {'input_dir': input_dir, "source_prefix": source_prefix, 'column_rename_dict': column_rename_dict}
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.input_dir = input_dir
        self.source_prefix = source_prefix
        self.column_rename_dict = column_rename_dict

    def get_files_with_subtask(self, file_type):
        from pyrecdp.primitives.llmutils.utils import get_target_file_list_from_local, sub_task_per_folder
        file_name = None
        if not os.path.isdir(self.input_dir):
            file_name = os.path.basename(self.input_dir)
            input_dir = os.path.dirname(self.input_dir)
        else:
            input_dir = self.input_dir
        files_with_subtask = sub_task_per_folder(get_target_file_list_from_local(self.input_dir, file_name if file_name is not None else file_type))
        logger.info(f"Load {files_with_subtask} as subtasks from {self.input_dir}")
        return files_with_subtask, input_dir

class SourcedJsonlReader(SourcedReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix, column_rename_dict = column_rename_dict)

    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        def add_source_str(content, source_str):
            content['source_id'] = source_str
            return content

        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            ds = self.rename_ray_ds_columns(rd.read_json(file_path).map(lambda x: add_source_str(x, os.path.join(self.source_prefix, sub_task, os.path.basename(file_path)))))
            ds_count = ds.count()
            self.statistics.total_in += ds_count
            self.statistics.total_out += ds_count
            self.cache = ds if idx == 0 else self.union_ray_ds(self.cache, ds)
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = self.rename_spark_df_columns(spark.read.json(file_path))
            df = df.withColumn('source_id', F.lit(os.path.join(self.source_prefix, sub_task, os.path.basename(file_path)))).cache()
            self.statistics.total_in += df.count()
            if '_corrupt_record' in df.columns:
                df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out += df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache

    def summarize(self) -> str:
        self.get_modified_rows()
        return (
            f"A total of {self.statistics.total_in} rows of json data were processed, "
            f"with {self.statistics.total_changed} rows read corrupted, {self.statistics.total_out} rows of data remaining.")

LLMOPERATORS.register(SourcedJsonlReader)

class GlobalJsonlReader(SourcedJsonlReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix, column_rename_dict = column_rename_dict)

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = self.rename_spark_df_columns(spark.read.json(file_path).cache())
            self.statistics.total_in += df.count()
            if '_corrupt_record' in df.columns:
                df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out += df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = df.select(F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias("global_id"), "*")
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache
LLMOPERATORS.register(GlobalJsonlReader)

class ParquetReader(TextReader):
    def __init__(self, input_dir = "", column_rename_dict = {}):
        settings = {'input_dir': input_dir, 'column_rename_dict': column_rename_dict}
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.input_dir = input_dir
        self.column_rename_dict = column_rename_dict

    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        self.cache = self.rename_ray_ds_columns(rd.read_parquet(self.input_dir))
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        self.cache = self.rename_spark_df_columns(spark.read.parquet(self.input_dir))
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache
LLMOPERATORS.register(ParquetReader)

class SourcedParquetReader(SourcedReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix, column_rename_dict = column_rename_dict)

    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        def add_source(s, source_str):
            s['source_id'] = source_str
            return s
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            ds = self.rename_ray_ds_columns(rd.read_parquet(file_path).map(lambda x: add_source(x, os.path.join(self.source_prefix, sub_task, os.path.basename(file_path)))))
            self.cache = ds if idx == 0 else self.union_ray_ds(self.cache, ds)
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = self.rename_spark_df_columns(spark.read.parquet(file_path))
            df = df.withColumn('source_id', F.lit(os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))))
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache
LLMOPERATORS.register(SourcedParquetReader)

class GlobalParquetReader(SourcedParquetReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix, column_rename_dict = column_rename_dict)

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        if spark_df:
            self.cache = spark_df
            return self.cache

        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = self.rename_spark_df_columns(spark.read.parquet(file_path))
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = df.select(
                F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias(
                    "global_id"), "*")
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache
LLMOPERATORS.register(GlobalParquetReader)


class PerfileReader:
    pass

class PerfileSourcedJsonlReader(SourcedReader, PerfileReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix, column_rename_dict = column_rename_dict)
        self.support_spark = True
        self.support_ray = True

    def process_rayds(self, ds=None):
        import ray.data as rd
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        def add_source_str(content, source_str):
            content['source_id'] = source_str
            return content

        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        self.cache = []
        for sub_task, file_path in to_read_list:
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            # ds = rd.read_text(file_path).map(lambda x: convert_json(x, source_id))
            ds = self.rename_ray_ds_columns(rd.read_json(file_path).map(lambda x: add_source_str(x, source_id)))
            ds_count = ds.count()
            self.statistics.total_in += ds_count
            self.statistics.total_out += ds_count
            self.cache.append((ds, source_id))
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None):
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        self.cache = []
        import pyspark.sql.functions as F

        for sub_task, file_path in to_read_list:
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = self.rename_spark_df_columns(spark.read.json(file_path))
            df = df.withColumn('source_id', F.lit(source_id)).cache()
            self.statistics.total_in += df.count()
            if '_corrupt_record' in df.columns:
                df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out += df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
            # if spark_df is not None, we need to add global_id for dataframe which will help to filter data with global_id
            if spark_df:
                df = df.select(F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias("global_id"), "*")
            self.cache.append((df, source_id))
        return self.cache

    def summarize(self) -> str:
        self.get_modified_rows()
        return (
            f"A total of {self.statistics.total_in} rows of json data were processed, "
            f"with {self.statistics.total_changed} rows read corrupted, {self.statistics.total_out} rows of data remaining.")

LLMOPERATORS.register(PerfileSourcedJsonlReader)

class PerfileSourcedParquetReader(SourcedReader, PerfileReader):
    def __init__(self, input_dir = "", source_prefix = "", column_rename_dict = {}):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix, column_rename_dict = column_rename_dict)
        self.support_spark = True
        self.support_ray = True

    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        def add_source(s, source_str):
            s['source_id'] = source_str
            return s
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        self.cache = []
        for sub_task, file_path in to_read_list:
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            ds = self.rename_ray_ds_columns(rd.read_parquet(file_path).map(lambda x: add_source(x, source_id)))
            self.cache.append((ds, source_id))
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None):
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        self.cache = []
        for sub_task, file_path in to_read_list:
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = self.rename_spark_df_columns(spark.read.parquet(file_path))
            # if spark_df is not None, we need to add global_id for dataframe which will help to filter data with global_id
            if spark_df:
                df = df.select(F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias("global_id"), "*")
            self.cache.append((df, source_id))
        return self.cache

LLMOPERATORS.register(PerfileSourcedParquetReader)
