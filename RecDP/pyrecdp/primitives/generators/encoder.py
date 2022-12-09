from autogluon.features.generators.label_encoder import LabelEncoderFeatureGenerator as SuperLabelEncoderFeatureGenerator
from autogluon.features.generators.one_hot_encoder import OneHotEncoderFeatureGenerator as SuperOneHotEncoderFeatureGenerator

class LabelEncoderFeatureGenerator(SuperLabelEncoderFeatureGenerator):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)


class OneHotEncoderFeatureGenerator(SuperOneHotEncoderFeatureGenerator):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)