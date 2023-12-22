"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import sys

from pyrecdp.core.model_utils import get_model, prepare_model
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.constant import HF_TOKENIZER
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
        settings = {'min_num': min_num, "max_num": max_num, "model_key": model_key}
        super().__init__(args_dict=settings)
        self.min_num = min_num
        self.max_num = max_num
        self.model_key = prepare_model(model_type='huggingface',
                                       model_key=model_key)
        self.tokenizer = get_model(self.model_key, model_type='huggingface')

    def get_compute_func(self, *args, **kwargs):
        min_num = self.min_num
        max_num = self.max_num
        tokenizer = self.tokenizer

        def compute(text) -> bool:
            tokens = get_words_from_document(
                text,
                token_func=tokenizer.tokenize if tokenizer else None
            )
            num_token = len(tokens)
            if min_num <= num_token <= max_num:
                return True
            else:
                return False
        return compute


LLMOPERATORS.register(TokenNumFilter)
