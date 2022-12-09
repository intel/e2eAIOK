from autogluon.features.generators.binned import BinnedFeatureGenerator as super_class

class BinnedFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
