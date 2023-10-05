from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame
from pyrecdp.core.model_utils import prepare_model, MODEL_ZOO
    
def prepare_func_sentencesplit(lang: str = 'en'):
    model_key = prepare_model(lang, model_type="nltk")
    nltk_model = MODEL_ZOO.get(model_key)
    tokenizer = nltk_model.tokenize if nltk_model else None
        
    def process(text):
        sentences = tokenizer(text)
        return '\n'.join(sentences)
    return process

class DocumentSplit(BaseLLMOperation):
    def __init__(self, text_key = 'text', inplace = True, language = 'en'):
        settings = {'text_key': text_key, 'inplace': inplace, 'language': language}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = inplace
        self.language = language
        self.actual_func = None
        self.support_spark = True
        self.support_ray = True
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'split_text'
        if self.actual_func is None:
            self.actual_func = prepare_func_sentencesplit(lang = self.language)
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, self.actual_func))
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'split_text'
        sentencesplit_udf = F.udf(prepare_func_sentencesplit(lang = self.language))
        return spark_df.withColumn(new_name, sentencesplit_udf(F.col(self.text_key)))
    
LLMOPERATORS.register(DocumentSplit)