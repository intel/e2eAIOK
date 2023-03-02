from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

class CategorifyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = op_base.config
    
    def get_function_pd(self):
        def categorify(df):
            for feature in self.feature_in:
                codes, uniques = pd.factorize(df[feature])
                df[f"{feature}__idx"] = pd.Series(codes, df[feature].index)
            return df
        return categorify
    


    def execute_spark(self, rdp):        
        raise NotImplementedError("Support later")