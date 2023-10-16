import sys

from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


# This filter is referred from alibaba data juicer project
class AverageLineLengthFilter(BaseFilter):
    def __init__(self, min_len=10, max_len=sys.maxsize):
        settings = {'min_len': min_len, "max_len": max_len}
        super().__init__(args_dict=settings)
        self.min_len = min_len
        self.max_len = max_len

    def compute(self, text) -> bool:
        lines = text.splitlines()
        avg_line_length = len(text) / len(lines) if len(lines) != 0 else 0.0
        if self.min_len <= avg_line_length <= self.max_len:
            return True
        else:
            return False

LLMOPERATORS.register(AverageLineLengthFilter)


