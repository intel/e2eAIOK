from .base import BaseFeatureGenerator as super_class

class DropDuplicatesFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return False
    
    def fit_prepare(self, pa_schema):
        return
    
    def get_function_pd(self):
        def drop_duplicates(df):
            return df
        return drop_duplicates