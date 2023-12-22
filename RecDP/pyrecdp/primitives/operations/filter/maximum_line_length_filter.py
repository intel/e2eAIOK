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

from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


# This filter is referred from alibaba data juicer project
class MaximumLineLengthFilter(BaseFilter):
    def __init__(self, min_len=10, max_len=sys.maxsize):
        """
            Keeps samples with maximum line length within the specified range

            :param min_len: The min filter length, samples will be filtered if their maximum line length is below this parameter. Default: 10
            :param max_len: The max filter length, samples will be filtered if their maximum line length exceeds this parameter. Default: sys.maxsize
        """
        settings = {'min_len': min_len, "max_len": max_len}
        super().__init__(args_dict=settings)
        self.min_len = min_len
        self.max_len = max_len

    def get_compute_func(self, *args, **kwargs):
        min_len = self.min_len
        max_len = self.max_len

        def compute(text) -> bool:
            lines = text.splitlines()
            line_lengths = list(map(len, lines))
            max_line_length = max(line_lengths) if line_lengths else 0.0
            if min_len <= max_line_length <= max_len:
                return True
            else:
                return False

        return compute


LLMOPERATORS.register(MaximumLineLengthFilter)
