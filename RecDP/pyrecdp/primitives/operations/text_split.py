from abc import ABC
from typing import List, Dict, Any, Optional, Callable

from pyspark.sql import DataFrame
from ray.data import Dataset

from pyrecdp.core.import_utils import import_langchain
from pyrecdp.core.model_utils import prepare_model
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


def prepare_text_split(text_splitter: str, **text_splitter_args) -> Callable[[str], List[str]]:
    if text_splitter == 'NLTKTextSplitter':
        """we have to download nltk model before we use it"""

        def prepare_nltk_model(model, lang):
            import nltk
            nltk.download('punkt')

        prepare_model(model_type="nltk", prepare_model_func=prepare_nltk_model)

    import_langchain()
    from pyrecdp.core.class_utils import new_instance
    from langchain.text_splitter import TextSplitter
    langchain_splitter: TextSplitter = new_instance("langchain.text_splitter", text_splitter,
                                                    **text_splitter_args)

    def process(text: str) -> List[str]:
        return langchain_splitter.split_text(text)

    return process


class DocumentSplit(BaseLLMOperation, ABC):
    def __init__(self, text_key: str = 'text',
                 inplace: bool = False,
                 text_splitter: Optional[str] = 'NLTKTextSplitter',
                 text_splitter_args: Optional[Dict] = None):

        """
        Args:
            text_key: The key of the text.
            inplace: Whether to operate on the original dataset.
            text_splitter(str): The class name of langchain text splitter.
            text_splitter_args: A dictionary of arguments to pass to the langchain text splitter.
        """
        text_splitter_args = text_splitter_args or {}
        settings = {
            'inplace': inplace,
            'text_key': text_key,
            'text_splitter': text_splitter,
            'text_splitter_args': text_splitter_args
        }
        super().__init__(settings)
        self.text_splitter = text_splitter
        self.text_splitter_args = text_splitter_args
        self.text_key = text_key
        self.support_ray = True
        self.support_spark = True
        if inplace:
            self.split_text_column = self.text_key
        else:
            self.split_text_column = 'split_text'

        self.text_split_func = prepare_text_split(self.text_splitter, **self.text_splitter_args)

    def process_rayds(self, ds: Dataset = None):
        def split_text(sample, text_split_func) -> List[Dict[str, Any]]:
            result = []
            for text in text_split_func(sample[self.text_key]):
                row = dict(**sample)
                row[self.split_text_column] = text
                result.append(row)
            return result

        return ds.flat_map(lambda sample: split_text(sample, self.text_split_func))

    def process_spark(self, spark, ds: DataFrame = None):
        import pyspark.sql.functions as F
        from pyspark.sql import types as T

        split_text_udf = F.udf(self.text_split_func, T.ArrayType(T.StringType()))
        ds = ds.withColumn(self.split_text_column, split_text_udf(F.col(self.text_key)))
        ds = ds.withColumn(self.split_text_column, F.explode(F.col(self.split_text_column)))
        return ds


LLMOPERATORS.register(DocumentSplit)
