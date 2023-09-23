from typing import Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import Row as SparkRow
from pyrecdp.models.model_utils import get_model
from pyrecdp.models.helper_func import get_sentences_from_document


class SentenceSplit:
    def __init__(self, text_key='text', lang: str = 'en', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        nltk_model = get_model(lang, model_type="nltk")
        self.tokenizer = nltk_model.tokenize if nltk_model else None

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get sentences from a document.

        :param sample: document that need to split sentences
        :return: document with the sentences separated by '\\\\n'
        """
        sample[self.text_key] = get_sentences_from_document(sample[self.text_key], model_func=self.tokenizer)
        return sample


def sentence_split(dataset: DataFrame) -> DataFrame:
    def do_split(batch):
        sentence_plit = SentenceSplit()
        for row in batch:
            row_dict = dict(**row.asDict())
            row_dict = sentence_plit.process(row_dict)
            yield SparkRow(**row_dict)

    return dataset.rdd.mapPartitions(do_split).toDF()
