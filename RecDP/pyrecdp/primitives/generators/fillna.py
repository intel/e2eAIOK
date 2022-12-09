from autogluon.features.generators.fillna import FillNaFeatureGenerator as super_class

class FillNaFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
