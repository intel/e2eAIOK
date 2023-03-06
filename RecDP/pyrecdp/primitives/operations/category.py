from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import copy

class CategorifyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = op_base.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self):
        feature_in = copy.deepcopy(self.feature_in)
        def categorify(df):
            for feature in feature_in:
                codes, uniques = pd.factorize(df[feature])
                df[f"{feature}__idx"] = pd.Series(codes, df[feature].index)
            return df
        return categorify