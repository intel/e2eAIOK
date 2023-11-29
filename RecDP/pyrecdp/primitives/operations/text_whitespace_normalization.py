from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

from pyrecdp.primitives.operations.constant import VARIOUS_WHITESPACES


class TextWhitespaceNormalization(BaseLLMOperation):
    def __init__(self, text_key='text'):
        """
            Normalize different kinds of whitespaces to whitespace ' ' (0x20) in text
            Different kinds of whitespaces can be found here:
                https://en.wikipedia.org/wiki/Whitespace_character
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
        def compute(text):
            # replace all kinds of whitespaces with ' '
            new_text = ''.join([
                char if char not in VARIOUS_WHITESPACES else ' ' for char in text
            ])
            return new_text

        return compute


LLMOPERATORS.register(TextWhitespaceNormalization)
