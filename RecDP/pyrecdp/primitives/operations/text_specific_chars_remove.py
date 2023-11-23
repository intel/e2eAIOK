from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

from typing import List, Union
import re


class TextSpecificCharsRemove(BaseLLMOperation):
    def __init__(self, text_key='text', chars_to_remove: Union[str, List[str]] = '◆●■►▼▲▴∆▻▷❖♡□'):
        settings = {'chars_to_remove': chars_to_remove, 'text_key': text_key}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
        self.text_key = text_key
        self.chars_to_remove = chars_to_remove
        self.inplace = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        remover = self.get_remover()
        return ds.map(lambda x: self.process_row(x, self.text_key, self.text_key, remover))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        custom_udf = F.udf(self.get_remover())
        return spark_df.withColumn(self.text_key, custom_udf(F.col(self.text_key)))

    def get_remover(self):
        pattern = '[' + '|'.join(self.chars_to_remove) + ']'

        def remover(text):
            text = re.sub(pattern=pattern, repl=r'',
                          string=text, flags=re.DOTALL)
            return text

        return remover


LLMOPERATORS.register(TextSpecificCharsRemove)
