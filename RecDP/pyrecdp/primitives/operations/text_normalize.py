from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
import ftfy, re, string

def normalize_str(s):
    s = ftfy.fix_text(s, normalization="NFC")
    return s

def clean_str(s):
    s = normalize_str(s)
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s.strip())
    return s

def text_normalization(s):
    return clean_str(s)

class TextNormalize(BaseLLMOperation):
    def __init__(self, text_key = 'text'):
        settings = {'text_key': text_key}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = False
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            raise NotImplementedError("We don't inplace modify text with normalization")
        else:
            new_name = 'norm_text'
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, text_normalization))
    
LLMOPERATORS.register(TextNormalize)