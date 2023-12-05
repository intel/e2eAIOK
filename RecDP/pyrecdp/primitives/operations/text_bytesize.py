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