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
    def __init__(self, text_key = 'text', inplace = True):
        self.text_key = text_key
        self.inplace = inplace
        settings = {'text_key': text_key, 'inplace': inplace}
        super().__init__(settings)
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'fixed_text'
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, text_normalization))
    
LLMOPERATORS.register(TextNormalize)