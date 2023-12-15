import math
from typing import List, Dict, Any, Optional, Callable, cast, Union

import numpy as np
from pyspark.sql import DataFrame
from ray.data import Dataset

from pyrecdp.core.import_utils import import_langchain, import_sentence_transformers, import_pysbd
from pyrecdp.core.model_utils import prepare_model
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


def prepare_text_split(text_splitter: Optional[str] = None, **text_splitter_args) -> Callable[[str], List[str]]:
    import_langchain()

    if 'NLTKTextSplitter' == text_splitter:
        """we have to download nltk model before we use it"""

        def prepare_nltk_model(model, lang):
            import nltk
            nltk.download('punkt')

        prepare_model(model_type="nltk", model_key="nltk_langchain", prepare_model_func=prepare_nltk_model)

    from pyrecdp.core.class_utils import new_instance
    splitter = new_instance("langchain.text_splitter", text_splitter, **text_splitter_args)

    def process(text: str) -> List[str]:
        return splitter.split_text(text)

    return process


class BaseDocumentSplit(BaseLLMOperation):
    def __init__(
            self,
            text_key: str = 'text',
            inplace: bool = True,
            args_dict: Optional[Dict] = None,
            requirements=[]
    ):
        settings = {
            'text_key': text_key,
            'inplace': inplace,
            'requirements': requirements,
        }
        settings.update(args_dict or {})
        requirements = requirements
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        self.text_key = text_key
        self.inplace = inplace
        self.text_split_func = self.get_text_split_func()
        if inplace:
            self.split_text_column = text_key
        else:
            self.split_text_column = 'split_text'

    def get_text_split_func(self) -> Callable[[str], List[str]]:
        """base interface to get split func"""

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


class DocumentSplit(BaseDocumentSplit):

    def __init__(
            self,
            text_key: str = 'text',
            inplace: bool = True,
            text_splitter: Optional[str] = 'NLTKTextSplitter',
            text_splitter_args: Optional[Dict] = None,
            requirements=[],
    ):
        """
        Args:
            text_key: The key of the text.
            text_splitter(str): The class name of langchain text splitter or a callable function.
            text_splitter_args: A dictionary of arguments to pass to the langchain text splitter.
        """
        if text_splitter is None:
            raise ValueError(f"text_splitter must be provide")

        text_splitter_args = text_splitter_args or {}
        self.text_splitter = text_splitter
        self.text_splitter_args = text_splitter_args
        settings = {
            'text_key': text_key,
            'text_splitter': text_splitter,
            'text_splitter_args': text_splitter_args,
            'requirements': requirements,
        }
        super().__init__(text_key=text_key,
                         inplace=inplace,
                         args_dict=settings,
                         requirements=requirements)

    def get_text_split_func(self) -> Callable[[str], List[str]]:
        return prepare_text_split(self.text_splitter,
                                  **self.text_splitter_args)


LLMOPERATORS.register(DocumentSplit)


# This text splitter is referred from
# https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6
def split_text_by_paragraphs(sentence_transformer, seg, paragraph_size: Optional[int] = 10):
    from sentence_transformers import SentenceTransformer
    from pysbd import Segmenter
    from scipy.signal import argrelextrema
    from sklearn.metrics.pairwise import cosine_similarity
    sentence_transformer = cast(SentenceTransformer, sentence_transformer)
    seg = cast(Segmenter, seg)

    def normalize_text(_text: str) -> List[str]:
        # We need to split whole text into sentences first.
        _sentences = seg.segment(_text)
        # Get the length of each sentence
        sentence_length = [len(each) for each in _sentences]
        # Determine longest outlier
        long = np.mean(sentence_length) + np.std(sentence_length) * 2
        # Determine shortest outlier
        short = np.mean(sentence_length) - np.std(sentence_length) * 2
        # Shorten long sentences
        _text = ''
        for each in _sentences:
            if len(each) > long:
                # let's replace all the commas with dots
                comma_splitted = each.replace(',', '.')
            else:
                _text += f'{each}. '
        _sentences = _text.split('. ')
        # Now let's concatenate short ones
        _text = ''
        for each in _sentences:
            if len(each) < short:
                _text += f'{each} '
            else:
                _text += f'{each}. '

        return _text.split('. ')

    def embed_sentences(_sentences: List[str]):
        _embeddings = sentence_transformer.encode(_sentences)
        # Normalize the embeddings
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
        return _embeddings / norms

    def rev_sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(0.5 * value))

    def activate_similarities(_similarities: np.array, p_size=10) -> np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            _similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space.
        # P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10, 10, p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid)
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect
        # of every additional sentence and to match the length of vector we will multiply
        pad_width = _similarities.shape[0] - p_size
        activation_weights = np.pad(y(x), (0, pad_width))

        # 1. Take each diagonal to the right of the main diagonal
        diagonals = [_similarities.diagonal(each) for each in range(0, _similarities.shape[0])]
        # 2. Pad each diagonal by zeros at the end.
        # Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0, _similarities.shape[0] - len(each))) for each in diagonals]
        # 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        # 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1, 1)
        # 5. Calculate the weighted sum of activated similarities
        return np.sum(diagonals, axis=0)

    def process(text: str) -> List[str]:
        # 1. We need to split whole text into sentences first.
        sentences = normalize_text(text)
        # 2. Embed sentences
        embeddings = embed_sentences(sentences)

        similarities = cosine_similarity(embeddings)
        # Let's apply our function. For long sentences i reccomend to use 10 or more sentences
        activated_similarities = activate_similarities(similarities, p_size=paragraph_size)

        # 3. Find relative minima of our vector.
        # For all local minimas and save them to variable with argrelextrema function
        minmimas = argrelextrema(activated_similarities, np.less, order=2)
        split_points = [each for each in minmimas[0]]
        text = ''
        paragraphs = []
        for num, each in enumerate(sentences):
            if num in split_points:
                paragraphs.append(text)
                text = f'\n\n {each}. '
            else:
                text += f'{each}. '

        paragraphs.append(text)

        return paragraphs

    return process


# This text splitter is referred from
# https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6
class ParagraphsTextSplitter(BaseDocumentSplit):
    def __init__(self,
                 text_key: Optional[str] = 'text',
                 inplace: bool = False,
                 model_name: Optional[str] = 'sentence-transformers/all-mpnet-base-v2',
                 language: Optional[str] = 'en',
                 paragraph_size: Optional[int] = 10,
                 requirements=[]):
        """
          Initializes the ParagraphsTextSplitter class.

          Args:
              text_key (Optional[str]): The key in the dataset to use as the text for embedding. Defaults to 'text'.
              model_name (Optional[str]): The name of the SentenceTransformer model to use. Defaults to 'sentence-transformers/all-mpnet-base-v2'.
              language (Optional[str]): The language of the text. Defaults to 'en'.
              paragraph_size (Optional[int]): The maximum number of sentences in a paragraph. Defaults to 10.
          """
        self.model_name = model_name
        self.p_size = paragraph_size
        self.language = language
        settings = {
            'model_name': model_name,
            'language': language,
            'paragraph_size': paragraph_size,
            'requirements': requirements,
        }
        super().__init__(text_key=text_key,
                         inplace=inplace,
                         args_dict=settings,
                         requirements=requirements)

    def get_text_split_func(self) -> Callable[[str], List[str]]:
        import_sentence_transformers()
        import_pysbd()
        from sentence_transformers import SentenceTransformer
        import pysbd
        model = SentenceTransformer(self.model_name)
        seg = pysbd.Segmenter(language=self.language, clean=False)
        return split_text_by_paragraphs(model, seg, self.p_size)


LLMOPERATORS.register(ParagraphsTextSplitter)


class CustomerDocumentSplit(BaseDocumentSplit):
    def __init__(
            self,
            func,
            inplace=True,
            text_key: str = 'text',
            requirements=[],
            **func_kwargs
    ):
        """
            Initialize the `CustomerDocumentSplit` class.

            Args:
                func: The Callable that will be used to split the text.
                inplace: Whether to perform split text in place.
                text_key: The key in the dictionary that contains the text to be tokenized.
                **func_kwargs: Keyword arguments to pass to the tokenization function.
        """
        if func is None:
            raise ValueError(f"func must be provide")
        if not callable(func):
            import os
            if not os.path.exists(func):
                raise FileNotFoundError(f'Reload {func} object but not exists')
            import pickle
            with open(func, 'rb') as f:
                self.split_func = pickle.load(f)
        else:
            self.split_func = func
        settings = {
            'func': func,
            'requirements': requirements,
        }
        settings.update(func_kwargs or {})
        self.func_kwargs = func_kwargs
        super().__init__(text_key=text_key,
                         inplace=inplace,
                         args_dict=settings,
                         requirements=requirements)

    def get_text_split_func(self) -> Callable[[str], List[str]]:
        def process(text):
            return self.split_func(text, **self.func_kwargs)

        return process


LLMOPERATORS.register(CustomerDocumentSplit)
