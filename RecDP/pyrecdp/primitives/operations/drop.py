from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD
 
class DropOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = op_base.config

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