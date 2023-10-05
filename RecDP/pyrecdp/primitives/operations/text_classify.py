from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset

def text_classify(s):
    return s

class Classify(BaseLLMOperation):
    def __init__(self, text_key = 'text', inplace = True):
        self.text_key = text_key
        self.inplace = inplace
        settings = {'text_key': text_key, 'inplace': inplace}
        super().__init__(settings)
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds.map(self.process_row)
        
    def process_row(self, sample: dict) -> dict:
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'norm_text'
        sample[new_name] = text_classify(sample[self.text_key])
        return sample
    
LLMOPERATORS.register(Classify)