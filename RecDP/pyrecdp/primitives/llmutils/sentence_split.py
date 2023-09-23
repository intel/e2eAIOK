from pyspark.sql import DataFrame
from pyspark.sql import Row as SparkRow
from pyrecdp.models.model_utils import get_model
from pyrecdp.models.helper_func import get_sentences_from_document


class SentenceSplit:
    def __init__(self, lang: str = 'en'):
        nltk_model = get_model(lang, model_type="nltk")
        self.tokenizer = nltk_model.tokenize if nltk_model else None

    def process(self, sample) -> str:
        """
        Get sentences from a document.

        :return: document with the sentences separated by '\\\\n'

        Args:
            sample: document that need to split sentences
        """
        return get_sentences_from_document(sample, model_func=self.tokenizer)


def sentence_split(dataset: DataFrame, text_column='text') -> DataFrame:
    def do_split(batch):
        sentence_plit = SentenceSplit()
        for row in batch:
            row_dict = dict(**row.asDict())
            row_dict[text_column] = sentence_plit.process(row_dict[text_column])
            yield SparkRow(**row_dict)

    return dataset.rdd.mapPartitions(do_split).toDF()
