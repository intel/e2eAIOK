import sys

from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.utils import get_words_from_document


# This filter is referred from alibaba data juicer project
class PerplexityFilter(BaseFilter):
    def __init__(self, language: str = 'en', max_ppl=1500):
        """
            Keeps samples with perplexity score below the specified threshold

            :param language: Sample in which language. Default: en.(en, zh)
            :param max_ppl: The max filter perplexity, samples will be filtered if their perplexity exceeds this parameter. Default: 1500
        """
        settings = {'language': language, "max_ppl": max_ppl}
        super().__init__(args_dict=settings)
        self.language = language
        self.max_ppl = max_ppl
        self.sp_model_key = prepare_model(lang=language,
                                          model_type='sentencepiece')
        self.kl_model_key = prepare_model(lang=language, model_type='kenlm')
        self.tokenizer = get_model(self.sp_model_key, self.language, 'sentencepiece')
        self.kenlm_model = get_model(self.kl_model_key, self.language, 'kenlm')

    def get_compute_func(self, *args, **kwargs):
        max_ppl = self.max_ppl
        tokenizer = self.tokenizer
        kenlm_model = self.kenlm_model

        def compute(text) -> bool:
            words = get_words_from_document(
                text,
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            join_text = ' '.join(words)
            # compute perplexity
            logits, length = 0, 0
            for line in join_text.splitlines():
                logits += kenlm_model.score(line)
                length += (len(line.split()) + 1)
            ppl = (10.0 ** (-logits / length)) if length != 0 else 0.0
            perplexity = round(ppl, 1)
            return perplexity <= max_ppl

        return compute


LLMOPERATORS.register(PerplexityFilter)
