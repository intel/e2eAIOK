from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


class LengthFilter(BaseFilter):
    def __init__(self, min_len=100, max_len=-1):
        settings = {'min_len': min_len, "max_len": max_len}
        super().__init__(args_dict=settings)
        self.min_len = min_len
        self.max_len = max_len

    def compute(self, text) -> bool:
        if len(text) < self.min_len or (self.max_len != -1 and len(text) > self.max_len):
            return False
        else:
            return True


LLMOPERATORS.register(LengthFilter)


