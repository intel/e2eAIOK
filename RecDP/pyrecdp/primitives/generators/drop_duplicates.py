from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator as super_class

class DropDuplicatesFeatureGenerator(super_class):
    def __init__(self, orig_generator = None, **kwargs):
        if orig_generator:
            self.obj = orig_generator
        else:
            self.obj = None
            super().__init__(**kwargs)
        
    def __getattr__(self, attr):
        if self.obj:
            return getattr(self.obj, attr)
        else:
            return getattr(self, attr)

    def _fit_transform(self, X, **kwargs):
        return super()._fit_transform(X, **kwargs)

    def is_useful(self, df):
        return True

    def update_feature_statistics(self, X, state_dict):
        return state_dict
