from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter
from profanity_check import predict


class ProfanityFilter(BaseFilter):
    def compute(self, text) -> bool:
        scores = predict([text])
        ret = not bool(scores[0])
        return ret


LLMOPERATORS.register(ProfanityFilter)
