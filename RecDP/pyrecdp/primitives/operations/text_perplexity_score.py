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

from .base import BaseLLMOperation, statistics_decorator
from ray.data import Dataset
from pyspark.sql import DataFrame
from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.utils import get_words_from_document


def text_bytesize(s):
    return len(s.encode('utf-8'))


class TextPerplexityScore(BaseLLMOperation):
    def __init__(self, text_key: str = 'text', language: str = 'en'):
        """
             Generate perplexity score

            :param language: Sample in which language. Default: en.(en, zh)
        """
        settings = {'language': language, 'text_key': text_key}
        requirements = []
        super().__init__(settings, requirements)
        self.language = language
        self.text_key = text_key
        self.inplace = False
        self.sp_model_key = prepare_model(lang=language,
                                          model_type='sentencepiece')
        self.kl_model_key = prepare_model(lang=language, model_type='kenlm')
        self.tokenizer = get_model(self.sp_model_key, self.language, 'sentencepiece')
        self.kenlm_model = get_model(self.kl_model_key, self.language, 'kenlm')

    @statistics_decorator
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            raise NotImplementedError("We don't inplace modify text with normalization")
        else:
            new_name = 'perplexity'
        compute_func = self.get_compute_func()
        ret = ds.map(lambda x: self.process_row(x, self.text_key, new_name, compute_func))
        if self.statistics_flag:
            self.statistics.max = ret.max(new_name)
            self.statistics.min = ret.min(new_name)
            self.statistics.mean = ret.mean(new_name)
            self.statistics.std = ret.std(new_name)
        else:
            self.statistics.max, self.statistics.min, self.statistics.mean, self.statistics.std = 0, 0, 0, 0
        return ret

    @statistics_decorator
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        bytesize_udf = F.udf(self.get_compute_func(), T.FloatType())
        ret = spark_df.withColumn("perplexity", bytesize_udf(F.col(self.text_key)))
        if self.statistics_flag:
            self.statistics.max = ret.select(F.max("perplexity")).collect()[0][0]
            self.statistics.min = ret.select(F.min("perplexity")).collect()[0][0]
            self.statistics.mean = ret.select(F.mean("perplexity")).collect()[0][0]
            self.statistics.std = ret.select(F.std("perplexity")).collect()[0][0]
        else:
            self.statistics.max, self.statistics.min, self.statistics.mean, self.statistics.std = 0, 0, 0, 0
        return ret

    def get_compute_func(self, *args, **kwargs):
        tokenizer = self.tokenizer
        kenlm_model = self.kenlm_model

        def compute(text):
            words = get_words_from_document(
                text,
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            join_text = ' '.join(words)
            # compute perplexity
            logits, length = 0, 0
            for line in join_text.splitlines():
                logits += kenlm_model.score(line)
                length += (len(line.split()) + 1)
            ppl = (10.0 ** (-logits / length)) if length != 0 else 0.0
            perplexity = round(ppl, 1)
            return perplexity

        return compute

    def summarize(self) -> str:
        statistics_save = {
            "min": self.statistics.min,
            "max": self.statistics.max,
            "mean": self.statistics.mean,
            "std": self.statistics.std,
        }
        return (statistics_save,
                f"A total of {self.statistics.total_in} rows of data were processed, using {self.statistics.used_time} seconds, "
                f"Get max perplexity {self.statistics.max}, "
                f"Get min perplexity {self.statistics.min}, "
                f"Get average perplexity {self.statistics.mean},"
                f"Get the std of perplexity {self.statistics.std}")


LLMOPERATORS.register(TextPerplexityScore)
