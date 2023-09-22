from pyrecdp.models.model_utils import prepare_model,MODEL_ZOO

from pyspark.sql import DataFrame
from pyspark.sql import Row as SparkRow


def get_sentences_from_document(document, model_func=None):
    if model_func:
        sentences = model_func(document)
    else:
        sentences = document.splitlines()
    return '\n'.join(sentences)


class SentenceSplit:
    def __init__(self, lang: str = 'en', text_key: str = 'text'):
        self.model_key = prepare_model(lang, model_type="nltk")
        self.text_key = text_key

    def process(self, sample):
        """
        Get sentences from a document.

        :param document: document that need to split sentences
        :param model_func: function of sentence model, if specified, the
            function will be used for spliting document into different
            sentences.
        :return: document with the sentences separated by '\\\\n'
        """
        nltk_model = MODEL_ZOO.get(self.model_key, None)
        sample[self.text_key] = get_sentences_from_document(sample[self.text_key],
                                                            model_func=nltk_model.tokenize if nltk_model else None)
        return sample


def sentence_split(dataset: DataFrame, text_column: str = "text", lang: str = 'en'):
    def do_split(batch):
        sentence_plit = SentenceSplit(text_key=text_column, lang=lang)
        for row in batch:
            row_dict = dict(**row.asDict())
            sentence_plit.process(row_dict)
            yield SparkRow(**row_dict)

    return dataset.rdd.mapPartitions(do_split).toDF()
