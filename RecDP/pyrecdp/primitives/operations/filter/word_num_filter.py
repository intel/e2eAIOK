import sys

from pyrecdp.core.model_utils import get_model
from pyrecdp.primitives.operations.utils import get_words_from_document, words_refinement
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.filter.constant import SPECIAL_CHARACTERS


# This filter is referred from alibaba data juicer project
class WordNumFilter(BaseFilter):
    def __init__(self, min_num=10, max_num=sys.maxsize, language='en'):
        """
            Keeps samples with word count within the specified range

            :param min_num: The min filter word number, samples will be filtered if their word number is below this parameter. Default: 10
            :param max_num: The max filter word number, samples will be filtered if their word number exceeds this parameter. Default: sys.maxsize
            :param language: Sample in which language. Default: en. (en, zh)
        """
        settings = {'min_num': min_num, "max_num": max_num, "language": language}
        super().__init__(args_dict=settings)
        self.min_num = min_num
        self.max_num = max_num
        self.language = language
        self.model_key = None

    def get_compute_func(self, *args, **kwargs):
        tokenizer = get_model(self.model_key, lang=self.language,
                              model_type='sentencepiece')
        min_num = self.min_num
        max_num = self.max_num

        def compute(text) -> bool:

            words = get_words_from_document(
                text, token_func=tokenizer.encode_as_pieces if tokenizer else None)

            words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
            num_words = len(words)
            if min_num <= num_words <= max_num:
                return True
            else:
                return False

        return compute


LLMOPERATORS.register(WordNumFilter)
