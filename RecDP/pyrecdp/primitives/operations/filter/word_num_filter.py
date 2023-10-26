import sys

from pyrecdp.core.model_utils import get_model
from pyrecdp.primitives.operations.utils import get_words_from_document, words_refinement
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.filter.constant import SPECIAL_CHARACTERS


# This filter is referred from alibaba data juicer project
class WordNumFilter(BaseFilter):
    def __init__(self, min_num=10, max_num=sys.maxsize, language='en'):
        settings = {'min_num': min_num, "max_num": max_num, "language": language}
        super().__init__(args_dict=settings)
        self.min_num = min_num
        self.max_num = max_num
        self.language = language
        self.model_key = None

    def compute(self, text) -> bool:

        tokenizer = get_model(self.model_key, lang=self.language,
                              model_type='sentencepiece')
        words = get_words_from_document(
            text, token_func=tokenizer.encode_as_pieces if tokenizer else None)

        words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
        num_words = len(words)
        if self.min_num <= num_words <= self.max_num:
            return True
        else:
            return False


LLMOPERATORS.register(WordNumFilter)
