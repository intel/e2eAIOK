from .base import BaseFeatureGenerator as super_class
from pyrecdp.core import SeriesSchema
from typing import List
    
class RenameFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renamed = {}
   
    def is_useful(self, pa_schema: List[SeriesSchema]):
        found = False
        for pa_field in pa_schema:
            if '.' in pa_field.name:
                self.renamed[pa_field.name] = pa_field.name.replace('.', '__')
                found = True
        return found
    
    def fit_prepare(self, pa_schema):
        ret_schema = []
        for pa_field in pa_schema:
            if pa_field.name in self.renamed:
                pa_field.name = self.renamed[pa_field.name]
            ret_schema.append(pa_field)
        return ret_schema

    def get_function_pd(self):
        def rename_feature(df):
            return df.rename(columns=self.renamed)
        return rename_feature