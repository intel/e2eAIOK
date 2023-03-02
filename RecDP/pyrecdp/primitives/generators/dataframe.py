from .base import BaseFeatureGenerator as super_class

class DataframeConvertFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        return pipeline, children[0], max_idx
    
class DataframeTransformFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        return pipeline, children[0], max_idx