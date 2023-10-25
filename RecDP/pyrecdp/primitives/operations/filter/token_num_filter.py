import sys

from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.filter.constant import HF_TOKENIZER
from pyrecdp.primitives.operations.utils import get_words_from_document


# This filter is referred from alibaba data juicer project
class TokenNumFilter(BaseFilter):
    def __init__(self, min_num=10, max_num=sys.maxsize, model_key=HF_TOKENIZER):
        """
            Keeps samples with token count within the specified range

            :param min_num: The min filter token number, samples will be filtered if their token number is below this parameter. Default: 10
            :param max_num: The max filter token number, samples will be filtered if their token number exceeds this parameter. Default: sys.maxsize
            :param model_key: The tokenizer name of Hugging Face tokenizers. Default: _EleutherAI/pythia-6.9b-deduped_
        """
        settings = {'min_num': min_num, "max_num": max_num}
        super().__init__(args_dict=settings)
        self.min_num = min_num
        self.max_num = max_num
        self.model_key = prepare_model(model_type='huggingface',
                                       model_key=model_key)
        self.tokenizer = get_model(self.model_key, model_type='huggingface')

    def compute(self, text) -> bool:

        tokens = get_words_from_document(
            text,
            token_func=self.tokenizer.tokenize if self.tokenizer else None
        )
        num_token = len(tokens)
        if self.min_num <= num_token <= self.max_num:
            return True
        else:
            return False


LLMOPERATORS.register(TokenNumFilter)
