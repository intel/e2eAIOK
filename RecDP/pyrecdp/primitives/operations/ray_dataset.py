from .base import BaseLLMOperation, LLMOPERATORS
import copy
from ray.data import Dataset

def read_data():
    return None

class RayDatasetReader(BaseLLMOperation):
    def __init__(self):
        super().__init__()
        
    def process_rayds(self) -> Dataset:
        if self.cache == None:
            self.cache = read_data()
        return self.cache
LLMOPERATORS.register(RayDatasetReader)


class RayDatasetWriter(BaseLLMOperation):
    def __init__(self):
        super().__init__()
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds
LLMOPERATORS.register(RayDatasetWriter)