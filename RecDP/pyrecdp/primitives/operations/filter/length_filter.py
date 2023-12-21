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

from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


class LengthFilter(BaseFilter):
    def __init__(self, min_len=100, max_len=-1):
        """
            Keeps samples with total text length within the specified range

            :param min_len: The min text length in the filtering. samples will be filtered if their text length is below this parameter. Default: 100
            :param max_len: The max text length in the filtering. samples will be filtered if their text length exceeds this parameter. Default: -1(unlimited)
        """
        settings = {'min_len': min_len, "max_len": max_len}
        super().__init__(args_dict=settings)
        self.min_len = min_len
        self.max_len = max_len

    def compute(self, text) -> bool:
        if len(text) < self.min_len or (self.max_len != -1 and len(text) > self.max_len):
            return False
        else:
            return True

    def get_compute_func(self, *args, **kwargs):
        min_len = self.min_len
        max_len = self.max_len

        def compute(text) -> bool:
            if text is None or len(text) < min_len or (max_len != -1 and len(text) > max_len):
                return False
            else:
                return True

        return compute


LLMOPERATORS.register(LengthFilter)
