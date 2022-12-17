from .base import BaseFeatureGenerator as super_class

class DataframeConvertFeatureGenerator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return False
    
    def fit_prepare(self, pa_schema):
        return

    def get_function_pd(self):
        def convert_df(df):
            return df
        return convert_df