from .base import BaseLLMOperation, LLMOPERATORS
import copy
from ray.data import Dataset
from pyspark.sql import DataFrame
import json

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
        
    def process_rayds(self, ds) -> Dataset:
        import ray.data as rd
        def convert_json(s):
            if isinstance(s, str):
                content = json.loads(s)
            elif isinstance(s, dict):
                content = {}
                for key in s:
                    content[key] = str(s[key])  
            return content
        self.cache = rd.read_text(self.input_dir).map(convert_json)
        return self.cache
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        from pyspark.sql.types import StructType, StructField, StringType
        import pyspark.sql.functions as F
        schema = StructType([ 
            StructField("text",StringType(),True), 
            StructField("meta",StringType(),True)
        ])
        df = spark.read.text(self.input_dir)            
        df = df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*")
        self.cache = df            
        return self.cache
LLMOPERATORS.register(JsonlReader)

class ParquetReader(BaseLLMOperation):
    def __init__(self, input_dir = ""):
        super().__init__()
        self.input_dir = input_dir
        settings = {'input_dir': input_dir}
        super().__init__(settings)        
        
    def process_rayds(self, ds) -> Dataset:
        import ray.data as rd
        self.cache = rd.read_parquet(self.input_dir)
        return self.cache
    
    def process_spark(self, spark, spark_df: DataFrame = None) -> DataFrame:
        df = spark.read.parquet(self.input_dir)
        self.cache = df            
        return self.cache
LLMOPERATORS.register(ParquetReader)

class RayDatasetWriter(BaseLLMOperation):
    def __init__(self):
        super().__init__()
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds
LLMOPERATORS.register(RayDatasetWriter)