from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD
import copy
 
class DropOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = self.op.config
        self.support_spark_dataframe = True
        self.support_spark_rdd = True
        self.fast_without_dpp = True

    def get_function_pd(self):
        feature_in = copy.deepcopy(self.feature_in)
        def drop_useless_feature(df):
            return df.drop(columns = feature_in)
        return drop_useless_feature
    
    def get_function_spark(self, rdp):
        def drop_feature(df):
            return df.drop(*self.feature_in)
        return drop_feature