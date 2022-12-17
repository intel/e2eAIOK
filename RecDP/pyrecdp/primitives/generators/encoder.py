from .base import BaseFeatureGenerator as super_class

class LabelEncoderFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return False
    
    def fit_prepare(self, pa_schema):
        return
    
    def get_function_pd(self):
        def label_encode(df):
            return df
        return label_encode


class OneHotEncoderFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return False
    
    def fit_prepare(self, pa_schema):
        return
    
    def get_function_pd(self):
        def onehot_encode(df):
            return df
        return onehot_encode