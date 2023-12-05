from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter


class ProfanityFilter(BaseFilter):
    def __init__(self, threshold: float = 0.0):
        """
            Keeps sample without profanity language. Mainly using alt-profanity-check library

            :param threshold: The max profanity threshold, samples will be filtered if their profanity score exceeds this parameter. Default: 0.0 (Float 0-1)
        """
        settings = {'threshold': threshold}
        super().__init__(args_dict=settings)
        self.threshold = threshold
        requirements = ['alt-profanity-check==1.3.0']
        super().__init__(settings, requirements)

    def get_compute_func(self, *args, **kwargs):
        from profanity_check import predict, predict_prob
        threshold = self.threshold
        predict_func = predict
        if self.threshold != 0:
            predict_func = predict_prob

        def compute(text) -> bool:
            scores = predict_func([text])
            if scores[0] <= threshold:
                return True
            else:
                return False
        return compute


LLMOPERATORS.register(ProfanityFilter)
