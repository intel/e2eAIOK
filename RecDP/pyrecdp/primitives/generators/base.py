from pyrecdp.core import SeriesSchema
from typing import List

class BaseFeatureGenerator:
    def __init__(self):
        pass
    
    def fit_prepare(self, pipeline, children, max_idx):
        return pipeline, children[0], max_idx