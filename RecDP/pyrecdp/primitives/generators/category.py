from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core import SeriesSchema
from typing import List
from pyspark.sql import DataFrame as SparkDataFrame

class CategoryFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_in = []

    def fit_prepare(self, pa_schema: List[SeriesSchema]):
        is_useful = False
        for pa_field in pa_schema:
            if pa_field.is_categorical_and_string:
                feature = pa_field.name
                self.feature_in.append(pa_field.name)
                is_useful = True
                pa_schema.append(SeriesSchema(f"{feature}__idx", pd.CategoricalDtype()))
        return pa_schema, is_useful
    
    def get_function_pd(self):
        def categorify(df):
            for feature in self.feature_in:
                codes, uniques = pd.factorize(df[feature])
                df[f"{feature}__idx"] = pd.Series(codes, df[feature].index)
            return df
        return categorify

    def get_function_spark(self, rdp):        
        actual_func = self.get_function_pd()
        def categorify(df):
            # check input df type
            if isinstance(df, pd.DataFrame):
                return actual_func(df)
            elif isinstance(df, SparkDataFrame):
                raise NotImplementedError("Support later")
        return categorify