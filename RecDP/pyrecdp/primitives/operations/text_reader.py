from .base import BaseLLMOperation, LLMOPERATORS
import copy
from ray.data import Dataset
from pyspark.sql import DataFrame
import json
import os

class DatasetReader(BaseLLMOperation):
    def __init__(self):
        super().__init__()
        self.support_ray = True
        self.support_spark = True

LLMOPERATORS.register(DatasetReader)

class JsonlReader(BaseLLMOperation):
    def __init__(self, input_dir = ""):
        settings = {'input_dir': input_dir}
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.input_dir = input_dir
        
    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        self.cache = rd.read_json(self.input_dir)
        self.statistics.total_in = self.cache.count()
        self.statistics.total_out = self.statistics.total_in
        return self.cache
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        df = spark.read.json(self.input_dir).cache()
        self.statistics.total_in = df.count()
        if '_corrupt_record' in df.columns:
            df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out = df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
        else:
            self.statistics.total_out = self.statistics.total_in
        self.cache = df            
        return self.cache

    def summarize(self) -> str:
        self.get_modified_rows()
        return (
            f"A total of {self.statistics.total_in} rows of json data were processed, "
            f"with {self.statistics.total_changed} rows read corrupted, {self.statistics.total_out} rows of data remaining.")

LLMOPERATORS.register(JsonlReader)

class SourcedReader(BaseLLMOperation):
    def __init__(self, input_dir = "", source_prefix = ""):
        settings = {'input_dir': input_dir, "source_prefix": source_prefix}
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.input_dir = input_dir
        self.source_prefix = source_prefix
        
    def get_files_with_subtask(self, file_type):
        from pyrecdp.primitives.llmutils.utils import get_target_file_list_from_local, sub_task_per_folder
        file_name = None
        if not os.path.isdir(self.input_dir):
            file_name = os.path.basename(self.input_dir)
            input_dir = os.path.dirname(self.input_dir)
        else:
            input_dir = self.input_dir
        files_with_subtask = sub_task_per_folder(get_target_file_list_from_local(self.input_dir, file_name if file_name is not None else file_type))
        return files_with_subtask, input_dir
    
class SourcedJsonlReader(SourcedReader):
    def __init__(self, input_dir = "", source_prefix = ""):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix)
        
    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        def add_source_str(content, source_str):
            content['source_id'] = source_str
            return content

        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            ds = rd.read_json(file_path).map(lambda x: add_source_str(x, os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))))
            ds_count = ds.count()
            self.statistics.total_in += ds_count
            self.statistics.total_out += ds_count
            self.cache = ds if idx == 0 else self.union_ray_ds(self.cache, ds)
        return self.cache

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = spark.read.json(file_path)
            df = df.withColumn('source_id', F.lit(os.path.join(self.source_prefix, sub_task, os.path.basename(file_path)))).cache()
            self.statistics.total_in += df.count()
            if '_corrupt_record' in df.columns:
                df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out += df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        return self.cache

    def summarize(self) -> str:
        self.get_modified_rows()
        return (
            f"A total of {self.statistics.total_in} rows of json data were processed, "
            f"with {self.statistics.total_changed} rows read corrupted, {self.statistics.total_out} rows of data remaining.")

LLMOPERATORS.register(SourcedJsonlReader)

class GlobalJsonlReader(SourcedJsonlReader):
    def __init__(self, input_dir = "", source_prefix = ""):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix)

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = spark.read.json(file_path).cache()
            self.statistics.total_in += df.count()
            if '_corrupt_record' in df.columns:
                df = df.filter("_corrupt_record is NULL").drop("_corrupt_record")
            self.statistics.total_out += df.count()
            self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = df.select(F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias("global_id"), "*")
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        return self.cache
LLMOPERATORS.register(GlobalJsonlReader)

class ParquetReader(BaseLLMOperation):
    def __init__(self, input_dir = ""):        
        settings = {'input_dir': input_dir}
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.input_dir = input_dir
        
    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        self.cache = rd.read_parquet(self.input_dir)
        return self.cache
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        df = spark.read.parquet(self.input_dir)
        self.cache = df            
        return self.cache
LLMOPERATORS.register(ParquetReader)

class SourcedParquetReader(SourcedReader):
    def __init__(self, input_dir = "", source_prefix = ""):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix)
        
    def process_rayds(self, ds=None) -> Dataset:
        import ray.data as rd
        def add_source(s, source_str):
            s['source_id'] = source_str  
            return s
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            ds = rd.read_parquet(file_path).map(lambda x: add_source(x, os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))))
            self.cache = ds if idx == 0 else self.union_ray_ds(self.cache, ds)
        return self.cache
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = spark.read.parquet(file_path)
            df = df.withColumn('source_id', F.lit(os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))))
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        return self.cache
LLMOPERATORS.register(SourcedParquetReader)

class GlobalParquetReader(SourcedParquetReader):
    def __init__(self, input_dir = "", source_prefix = ""):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix)

    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        if spark_df:
            self.cache = spark_df
            return self.cache

        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("parquet")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        for idx, (sub_task, file_path) in enumerate(to_read_list):
            df = spark.read.parquet(file_path)
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = df.select(
                F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias(
                    "global_id"), "*")
            self.cache = df if idx == 0 else self.union_spark_df(self.cache, df)
        return self.cache
LLMOPERATORS.register(GlobalParquetReader)


class PerfileReader:
    pass

class PerfileSourcedJsonlReader(SourcedReader, PerfileReader):
    def __init__(self, input_dir = "", source_prefix = ""):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix)
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
            ds = rd.read_json(file_path).map(lambda x: add_source_str(x, source_id))
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
            df = spark.read.json(file_path)
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
    def __init__(self, input_dir = "", source_prefix = ""):
        super().__init__(input_dir = input_dir, source_prefix = source_prefix)
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
            ds = rd.read_parquet(file_path).map(lambda x: add_source(x, source_id))
            self.cache.append((ds, source_id))
        return self.cache
    
    def process_spark(self, spark, spark_df: DataFrame = None):
        import pyspark.sql.functions as F
        files_with_subtask, input_dir = self.get_files_with_subtask("jsonl")
        to_read_list = [(sub_task, os.path.join(input_dir, f)) for sub_task, file_list in files_with_subtask.items() for f in file_list]
        self.cache = []
        for sub_task, file_path in to_read_list:
            source_id = os.path.join(self.source_prefix, sub_task, os.path.basename(file_path))
            df = spark.read.parquet(file_path)
            # if spark_df is not None, we need to add global_id for dataframe which will help to filter data with global_id
            if spark_df:
                df = df.select(F.concat_ws("@", F.monotonically_increasing_id(), F.lit(source_id)).alias("global_id"), "*")
            self.cache.append((df, source_id))
        return self.cache
    
LLMOPERATORS.register(PerfileSourcedParquetReader)
