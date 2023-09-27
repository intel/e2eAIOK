from pyspark.sql import DataFrame
from pyrecdp.core.model_utils import prepare_model, MODEL_ZOO
from pyspark.sql.functions import udf


class SentenceSplit:
    def __init__(self, lang: str = 'en'):
        try:
            model_key = prepare_model(lang, model_type="nltk")
            nltk_model = MODEL_ZOO.get(model_key)
            self.tokenizer = nltk_model.tokenize if nltk_model else None
        except:
            self.tokenizer = None

    def process(self, sample) -> str:
        """
        Get sentences from a document.

        :return: document with the sentences separated by '\\\\n'

        Args:
            sample: document that need to split sentences
        """
        sentences = self.tokenizer(sample)
        return '\n'.join(sentences)


def sentence_split(dataset: DataFrame, text_column='text', new_text_column='text') -> DataFrame:
    sentence_plit = SentenceSplit()
    sentence_split_udf = udf(lambda sample: sentence_plit.process(sample))

    return dataset.withColumn(new_text_column, sentence_split_udf(text_column))
