from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame



class TextUnicodeFixer(BaseLLMOperation):
    def __init__(self, text_key='text'):
        """
            Fix unicode errors in text using ftfy
        """
        settings = {'text_key': text_key}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
        self.text_key = text_key
        self.inplace = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        remover = self.get_compute_func()
        return ds.map(lambda x: self.process_row(x, self.text_key, self.text_key, remover))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        custom_udf = F.udf(self.get_compute_func())
        return spark_df.withColumn(self.text_key, custom_udf(F.col(self.text_key)))

    def get_compute_func(self):
        import ftfy

        def compute(text):
            new_text = ftfy.fix_text(text)
            return new_text
        return compute


LLMOPERATORS.register(TextUnicodeFixer)
