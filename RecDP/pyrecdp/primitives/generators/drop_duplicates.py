from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator as super_class

class DropDuplicatesFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
