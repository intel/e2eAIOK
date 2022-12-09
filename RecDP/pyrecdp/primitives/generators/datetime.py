from autogluon.features.generators.datetime import DatetimeFeatureGenerator as super_class

class DatetimeFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)
