from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


# This filter is referred from alibaba data juicer project
class SpecialCharactersFilter(BaseFilter):
    def __init__(self, min_ratio=0.0, max_ratio=0.25):
        """
            Keeps samples with special-char ratio within the specified range

            :param min_ratio: The min filter ratio, samples will be filtered if their special-char ratio is below this parameter. Default: 0.0
            :param max_ratio: The max filter ratio, samples will be filtered if their special-char ratio exceeds this parameter. Default: 0.25
        """
        settings = {'min_ratio': min_ratio, "max_ratio": max_ratio}
        super().__init__(args_dict=settings)
        self.min_ratio = max_ratio
        self.max_ratio = max_ratio

    def get_compute_func(self, *args, **kwargs):
        from pyrecdp.primitives.operations.constant import SPECIAL_CHARACTERS
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio

        def compute(text) -> bool:
            special_char_ratio = (
                    len([c
                         for c in text if c in SPECIAL_CHARACTERS.value]) /
                    len(text)) if len(text) != 0 else 0.0
            if min_ratio <= special_char_ratio <= max_ratio:
                return True
            else:
                return False
        return compute

LLMOPERATORS.register(SpecialCharactersFilter)
