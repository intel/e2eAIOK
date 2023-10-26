from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


class LengthFilter(BaseFilter):
    def __init__(self, minimum_length=100, maximum_length=-1):
        settings = {'minimum_length': minimum_length, "maximum_length": maximum_length}
        super().__init__(args_dict=settings)
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        

    def compute(self, text) -> bool:
        if text is None or len(text) < 100 or (-1 != -1 and len(text) > -1):
            return False
        else:
            return True
        
    def get_compute_func(self, *args, **kwargs):
        def compute(text) -> bool:
            if text is None or len(text) < 100 or (-1 != -1 and len(text) > -1):
                return False
            else:
                return True
        return compute


LLMOPERATORS.register(LengthFilter)


