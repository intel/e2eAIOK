from .base import BaseLLMOperation, LLMOPERATORS
import copy
from ray.data import Dataset
import ray.data as rd
import json

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

class JsonlReader(BaseLLMOperation):
    def __init__(self, input_dir = ""):
        super().__init__()
        self.input_dir = input_dir
        settings = {'input_dir': input_dir}
        super().__init__(settings)
        
    def process_rayds(self, ds) -> Dataset:
        def convert_json(s):
            if isinstance(s, str):
                content = json.loads(s)
            elif isinstance(s, dict):
                content = {}
                for key in s:
                    content[key] = str(s[key])  
            return content
        self.cache = rd.read_text(self.input_dir).map(convert_json)
        return self.cache
LLMOPERATORS.register(JsonlReader)

class ParquetReader(BaseLLMOperation):
    def __init__(self, input_dir = ""):
        super().__init__()
        self.input_dir = input_dir
        settings = {'input_dir': input_dir}
        super().__init__(settings)        
        
    def process_rayds(self, ds) -> Dataset:
        self.cache = rd.read_parquet(self.input_dir)
        return self.cache
LLMOPERATORS.register(ParquetReader)

class RayDatasetWriter(BaseLLMOperation):
    def __init__(self):
        super().__init__()
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds
LLMOPERATORS.register(RayDatasetWriter)