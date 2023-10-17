import sys

from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.filter.constant import SPECIAL_CHARACTERS


# This filter is referred from alibaba data juicer project
class SpecialCharactersFilter(BaseFilter):
    def __init__(self, min_ratio=0.0, max_ratio=0.25):
        settings = {'min_ratio': min_ratio, "max_ratio": max_ratio}
        super().__init__(args_dict=settings)
        self.min_ratio = max_ratio
        self.max_ratio = max_ratio

    def compute(self, text) -> bool:
        special_char_ratio = (
                len([c
                     for c in text if c in SPECIAL_CHARACTERS]) /
                len(text)) if len(text) != 0 else 0.0
        if self.min_ratio <= special_char_ratio <= self.max_ratio:
            return True
        else:
            return False


LLMOPERATORS.register(SpecialCharactersFilter)
