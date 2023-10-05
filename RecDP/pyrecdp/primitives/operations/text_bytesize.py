from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset

def text_bytesize(s):
    return len(s.encode('utf-8'))

class TextBytesize(BaseLLMOperation):
    def __init__(self, text_key = 'text'):
        settings = {'text_key': text_key}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = False
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            raise NotImplementedError("We don't inplace modify text with normalization")
        else:
            new_name = 'bytesize'
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, text_bytesize))
    
LLMOPERATORS.register(TextBytesize)