from .base import BaseFeatureGenerator as super_class
from pyrecdp.core import SeriesSchema
from typing import List
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD
 
class DropUselessFeatureGenerator(super_class):
    def __init__(self, final = False, **kwargs):
        super().__init__(**kwargs)
        self._astype_feature_map = None
        self.feature_in = []
        self.final = final
   
    def is_useful(self, pa_schema: List[SeriesSchema]):
        found = False
        for pa_field in pa_schema:
            if not self.final:
                if not (pa_field.is_numeric or pa_field.is_categorical):
                    self.feature_in.append(pa_field.name)
                    print(f"{pa_field} should drop")
                    found = True
            else:
                if not (pa_field.is_numeric):
                    self.feature_in.append(pa_field.name)
                    print(f"{pa_field} should drop")
                    found = True
        return found
    
    def fit_prepare(self, pa_schema: List[SeriesSchema]):
        ret_schema = []
        for pa_field in pa_schema:
            if pa_field.name not in self.feature_in:
                ret_schema.append(pa_field)
        return ret_schema

    def get_function_pd(self):
        def drop_useless_feature(df):
            return df.drop(columns = self.feature_in)
        return drop_useless_feature
    
    def get_function_spark(self, rdp):        
        actual_func = self.get_function_pd()
        def transform(iter, *args):
            for x in iter:
                yield actual_func(x[0], *args), x[1]
        def drop_useless_feature(df):
            # check input df type
            if isinstance(df, pd.DataFrame):
                return actual_func(df)
            elif isinstance(df, RDD):
                return df.mapPartitions(transform)
            elif isinstance(df, SparkDataFrame):
                raise NotImplementedError("Support later")
        return drop_useless_feature