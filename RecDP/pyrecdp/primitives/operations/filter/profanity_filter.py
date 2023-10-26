from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from profanity_check import predict, predict_prob


class ProfanityFilter(BaseFilter):
    def __init__(self, threshold: float = 0.0):
        settings = {'threshold': threshold}
        super().__init__(args_dict=settings)
        self.threshold = threshold
        if self.threshold == 0:
            self.predict_func = predict
        else:
            self.predict_func = predict_prob

    def compute(self, text) -> bool:

        scores = self.predict_func([text])
        if scores[0] <= self.threshold:
            return True
        else:
            return False


LLMOPERATORS.register(ProfanityFilter)
