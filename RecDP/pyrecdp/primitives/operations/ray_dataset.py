from .base import BaseRayOperation, RAYOPERATORS
import copy
from ray.data import Dataset

def read_data():
    return None

class RayDatasetReader(BaseRayOperation):
    def __init__(self):
        super().__init__()
        
    def process_rayds(self) -> Dataset:
        if self.cache == None:
            self.cache = read_data()
        return self.cache
RAYOPERATORS.register(RayDatasetReader)


class RayDatasetWriter(BaseRayOperation):
    def __init__(self):
        super().__init__()
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds
RAYOPERATORS.register(RayDatasetWriter)