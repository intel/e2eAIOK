from .base import BaseLLMOperation, LLMOPERATORS, statistics_decorator
from ray.data import Dataset
from pyspark.sql import DataFrame
from .filter import BaseFilter

def text_bytesize(s):
    return len(s.encode('utf-8'))

class TextCustomerMap(BaseLLMOperation):
    def __init__(self, func, text_key = 'text'):
        settings = {'func': func, 'text_key': text_key}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
        self.func = func
        self.text_key = text_key
        self.new_key = f"{func.__name__}_text"
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds.map(lambda x: self.process_row(x, self.text_key, self.new_key, self.func))
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        custom_udf = F.udf(self.func)
        return spark_df.withColumn(self.new_key, custom_udf(F.col(self.text_key)))
    
LLMOPERATORS.register(TextCustomerMap)

class TextCustomerFilter(BaseFilter):
    def __init__(self, func, text_key = 'text'):
        settings = {'func': func, 'text_key': text_key}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
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