from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyrecdp.core.utils import increment_encoder
import copy

class CategorifyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = op_base.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
    
    def get_function_pd(self):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        def categorify(df):
            from sklearn.preprocessing import LabelEncoder
            import numpy
            import os
            for feature, (dict_path, feature_out) in feature_in_out.items():
                encoder = LabelEncoder()
                encoder.fit(df[feature])
                encoder = increment_encoder(encoder, dict_path)
                df[f"{feature_out}"] = pd.Series(encoder.transform(df[feature]))
                
            return df
        return categorify