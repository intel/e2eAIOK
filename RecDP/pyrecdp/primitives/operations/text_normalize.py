from .base import BaseRayOperation, RAYOPERATORS
import copy
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

class TextNormalize(BaseRayOperation):
    def __init__(self, text_key = 'text'):
        self.text_key = text_key
        super().__init__({'text_key': text_key})
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds.map(self.process_row)
        
    def process_row(self, sample: dict) -> dict:
        new_name = 'norm_text'
        sample[new_name] = text_normalization(sample[self.text_key])
        return sample
    
RAYOPERATORS.register(TextNormalize)