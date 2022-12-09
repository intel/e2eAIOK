from autogluon.features.generators.drop_unique import DropUniqueFeatureGenerator as super_class

class DropUniqueFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
