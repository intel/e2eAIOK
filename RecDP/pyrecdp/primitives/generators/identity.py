from autogluon.features.generators.identity import IdentityFeatureGenerator as super_class


class IdentityFeatureGenerator(super_class):
    def __init__(self, orig_generator):
        self.obj = orig_generator
        
    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def _fit_transform(self, X, **kwargs):
        return super()._fit_transform(X, **kwargs)
