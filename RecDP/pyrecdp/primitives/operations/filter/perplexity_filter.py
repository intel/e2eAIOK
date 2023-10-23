import sys

from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.utils import get_words_from_document


# This filter is referred from alibaba data juicer project
class PerplexityFilter(BaseFilter):
    def __init__(self, language: str = 'en', max_ppl=1500):
        settings = {'language': language, "max_ppl": max_ppl}
        super().__init__(args_dict=settings)
        self.language = language
        self.max_ppl = max_ppl
        self.sp_model_key = prepare_model(lang=language,
                                          model_type='sentencepiece')
        self.kl_model_key = prepare_model(lang=language, model_type='kenlm')
        self.tokenizer = get_model(self.sp_model_key, self.language, 'sentencepiece')
        self.kenlm_model = get_model(self.kl_model_key, self.language, 'kenlm')

    def compute(self, text) -> bool:
        words = get_words_from_document(
            text,
            token_func=self.tokenizer.encode_as_pieces if self.tokenizer else None)
        join_text = ' '.join(words)
        # compute perplexity
        logits, length = 0, 0
        for line in join_text.splitlines():
            logits += self.kenlm_model.score(line)
            length += (len(line.split()) + 1)
        ppl = (10.0 ** (-logits / length)) if length != 0 else 0.0
        perplexity = round(ppl, 1)
        return perplexity <= self.max_ppl


LLMOPERATORS.register(PerplexityFilter)
