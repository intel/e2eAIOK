import sys

from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


# This filter is referred from alibaba data juicer project
class AlphanumericFilter(BaseFilter):
    def __init__(self, min_ratio=0.25, max_ratio=sys.maxsize):
        """ Keeps samples with alphanumeric ratio within the specified range

            :param min_ratio: The min filter ratio, samples will be filtered if their alphabet/numeric ratio is below this parameter. Default: 0.25
            :param max_ratio: The max filter ratio, samples will be filtered if their alphabet/numeric ratio exceeds this parameter. Default: sys.maxsize
        """
        settings = {'min_ratio': min_ratio, "max_ratio": max_ratio}
        super().__init__(args_dict=settings)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def get_compute_func(self):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio

        def compute(text) -> bool:
            alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, text))
            alnum_ratio = (alnum_count / len(text)) if len(text) != 0 else 0.0
            if min_ratio <= alnum_ratio <= max_ratio:
                return True
            else:
                return False

        return compute


LLMOPERATORS.register(AlphanumericFilter)
