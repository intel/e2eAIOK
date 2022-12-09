from autogluon.features.generators.category import CategoryFeatureGenerator as super_class

class CategoryFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
