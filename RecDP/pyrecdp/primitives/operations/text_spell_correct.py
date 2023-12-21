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

from ray.data import Dataset
from pyspark.sql import DataFrame

from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


class TextSpellCorrect(BaseLLMOperation):
    def __init__(self, text_key='text', inplace: bool = False):
        """
            Spelling correction for text using library [textblog](https://textblob.readthedocs.io/en/dev/)

        """
        settings = {'text_key': text_key, 'inplace': inplace}
        requirements = ["textblob"]
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        self.text_key = text_key

    def process_rayds(self, ds: Dataset) -> Dataset:
        spell_corrector = self.get_compute_func()
        return ds.map(lambda x: self.process_row(x, self.text_key, self.text_key, spell_corrector))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        custom_udf = F.udf(self.get_compute_func())
        return spark_df.withColumn(self.text_key, custom_udf(F.col(self.text_key)))

    def get_compute_func(self):
        from textblob import TextBlob

        def spell_corrector(text):
            return str(TextBlob(text).correct())

        return spell_corrector


LLMOPERATORS.register(TextSpellCorrect)
