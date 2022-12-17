from .base import BaseFeatureGenerator as super_class

class CategoryFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return False
    
    def fit_prepare(self, pa_schema):
        return
    
    def get_function_pd(self):
        def categorify(df):
            return df
        return categorify