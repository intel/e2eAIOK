from autogluon.features.generators.astype import AsTypeFeatureGenerator as super_class

class AsTypeFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
