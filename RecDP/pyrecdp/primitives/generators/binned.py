from .base import BaseFeatureGenerator as super_class

class BinnedFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pa_schema):
        return pa_schema, False
    
    def get_function_pd(self):
        def generate_bin(df):
            return df
        return generate_bin