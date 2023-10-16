import sys

from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


# This filter is referred from alibaba data juicer project
class AlphanumericFilter(BaseFilter):
    def __init__(self, min_ratio=0.25, max_ratio=sys.maxsize):
        settings = {'min_ratio': min_ratio, "max_ratio": max_ratio}
        super().__init__(args_dict=settings)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def compute(self, text) -> bool:
        alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, text))
        alnum_ratio = (alnum_count / len(text)) if len(text) != 0 else 0.0
        if self.min_ratio <= alnum_ratio <= self.max_ratio:
            return True
        else:
            return False

LLMOPERATORS.register(AlphanumericFilter)


