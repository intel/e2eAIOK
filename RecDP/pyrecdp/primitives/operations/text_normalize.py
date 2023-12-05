from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

def normalize_str(s):
    import ftfy
    s = ftfy.fix_text(s, normalization="NFC")
    return s

def clean_str(s):
    import string
    import re
    try:
        s = normalize_str(s)
    except:
        s = ""
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s.strip())
    return s

def text_normalization(s):
    return clean_str(s)

class TextNormalize(BaseLLMOperation):
    def __init__(self, text_key = 'text'):
        settings = {'text_key': text_key}
        requirements = ['ftfy']
        super().__init__(settings, requirements)
        self.text_key = text_key
        self.inplace = False
        self.support_spark = True
        self.support_ray = True
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            raise NotImplementedError("We don't inplace modify text with normalization")
        else:
            new_name = 'norm_text'
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, text_normalization))
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        new_name = 'norm_text'
        text_norm_udf = F.udf(text_normalization)
        return spark_df.withColumn(new_name, text_norm_udf(F.col(self.text_key)))
    
LLMOPERATORS.register(TextNormalize)