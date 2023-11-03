from .base import BaseLLMOperation
from ray.data import Dataset
from pyspark.sql import DataFrame
from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.utils import get_words_from_document


def text_bytesize(s):
    return len(s.encode('utf-8'))


class TextPerplexityScore(BaseLLMOperation):
    def __init__(self, language: str = 'en'):
        """
             Generate perplexity score

            :param language: Sample in which language. Default: en.(en, zh)
        """
        settings = {'language': language}
        super().__init__(args_dict=settings)
        self.language = language
        self.text_key = 'text'
        self.inplace = False
        self.sp_model_key = prepare_model(lang=language,
                                          model_type='sentencepiece')
        self.kl_model_key = prepare_model(lang=language, model_type='kenlm')
        self.tokenizer = get_model(self.sp_model_key, self.language, 'sentencepiece')
        self.kenlm_model = get_model(self.kl_model_key, self.language, 'kenlm')

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            raise NotImplementedError("We don't inplace modify text with normalization")
        else:
            new_name = 'perplexity'
        compute_func = self.get_compute_func()
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, compute_func))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        bytesize_udf = F.udf(self.get_compute_func(), T.FloatType())
        return spark_df.withColumn("perplexity", bytesize_udf(F.col(self.text_key)))

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


LLMOPERATORS.register(TextPerplexityScore)
