from .base import BaseFeatureGenerator as super_class
import pyarrow.types as types

class DatetimeFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []

    def is_useful(self, pa_schema):
        found = False
        for pa_field in pa_schema:
            if types.is_date32(pa_field.type) or types.is_timestamp(pa_field.type):
                self.feature_in.append(pa_field.name)
                found = True
        return found
    
    def fit_prepare(self, pa_schema):
        return pa_schema

    def get_function_pd(self):
        def generate_datetime_feature(df):
            return df
        return generate_datetime_feature