from pyrecdp.core.model_utils import prepare_model, get_model
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

import re


class TextSentenceResplit(BaseLLMOperation):
    def __init__(self, text_key='text', language: str = 'en'):
        """
            Re segment sentences in the text to avoid sentence segmentation errors caused by unnecessary line breaks

            :param language: Supported language. Default: en. (en)

        """
        settings = {'language': language, 'text_key': text_key}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
        self.text_key = text_key
        self.language = language
        self.inplace = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        remover = self.get_compute_func()
        return ds.map(lambda x: self.process_row(x, self.text_key, self.text_key, remover))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        custom_udf = F.udf(self.get_compute_func())
        return spark_df.withColumn(self.text_key, custom_udf(F.col(self.text_key)))

    def get_compute_func(self):
        model_key = prepare_model(lang=self.language, model_type='nltk')
        nltk_model = get_model(model_key, lang=self.language, model_type='nltk')

        def compute(text):
            pattern = "\\n\s*\\n"
            replace_str = '*^*^*'
            text = re.sub(pattern=pattern, repl=replace_str,
                              string=text, flags=re.DOTALL)

            sentences = nltk_model.tokenize(text)
            new_sentences = []
            for sentence in sentences:
                new_sentences.append(sentence.replace("\n", " "))
            new_text = ' '.join(new_sentences).replace(replace_str, "\n\n")
            return new_text

        return compute


LLMOPERATORS.register(TextSentenceResplit)
