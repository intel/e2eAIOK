import sys

from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.utils import get_words_from_document, words_refinement
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.filter.constant import SPECIAL_CHARACTERS


# This filter is referred from alibaba data juicer project
class WordRepetitionFilter(BaseFilter):
    def __init__(self,
                 language: str = 'en',
                 rep_len=10,
                 min_ratio=0.0,
                 max_ratio=0.5, ):
        settings = {'language': language, "rep_len": rep_len, "min_ratio": min_ratio,
                    "max_ratio": max_ratio}
        super().__init__(args_dict=settings)
        self.language = language
        self.n = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.model_key = prepare_model(lang=language,
                                       model_type='sentencepiece')
        self.tokenizer = get_model(self.model_key, lang=self.language,
                                   model_type='sentencepiece')

    def compute(self, text) -> bool:
        words = get_words_from_document(
            text,
            token_func=self.tokenizer.encode_as_pieces if self.tokenizer else None)
        words = words_refinement(
            words,
            lower_case=True,
            strip_chars=SPECIAL_CHARACTERS)
        word_ngrams = [
            ' '.join(words[i:i + self.n])
            for i in range(len(words) - self.n + 1)
        ]
        freq_word_ngrams = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = (
                    freq_word_ngrams.get(word_ngram, 0) + 1)
        if len(freq_word_ngrams) == 0:
            word_rep_ratio = 0.0
        else:
            freq_word_ngrams = list(freq_word_ngrams.values())
            rep_more_than_one = [freq for freq in freq_word_ngrams if freq > 1]
            word_rep_ratio = (sum(rep_more_than_one) /
                              sum(freq_word_ngrams)) if sum(freq_word_ngrams) != 0 else 0.0
        if self.min_ratio <= word_rep_ratio <= self.max_ratio:
            return True
        else:
            return False


LLMOPERATORS.register(WordRepetitionFilter)
