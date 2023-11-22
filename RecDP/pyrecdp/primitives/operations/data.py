from .base import BaseOperation, AUTOFEOPERATORS
import pandas as pd

class DataFrameOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
        self.fast_without_dpp = True

    def set(self, dataset):
        # this condition is very important, so place holder dataframe won't copy data
        if self.op.children is None or len(self.op.children) == 0:
            self.cache = dataset[self.op.config]
        
    def get_function_pd(self, trans_type = 'fit_transform'):
        cache = self.cache.copy() if self.cache is not None else None
        def get_dataframe(df):
            if df is not None:
                return df
            else:
                return cache
        return get_dataframe

AUTOFEOPERATORS.register(DataFrameOperation, "DataFrame")
    
class DataLoader(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.support_spark_dataframe = True
        self.support_spark_rdd = False
        
    def get_function_pd(self, trans_type = 'fit_transform'):
        def get_dataframe():
            file_path = self.op.config['file_path']
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            else:
                raise NotImplementedError("now sample read only support csv and parquet")
        return get_dataframe
    
    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        def get_dataframe():
            file_path = self.op.config['file_path']
            if file_path.endswith('.csv'):
                return rdp.spark.read.csv(file_path, header=True, inferSchema=True)
            elif file_path.endswith('.parquet'):
                return rdp.spark.read.parquet(file_path)
            else:
                raise NotImplementedError("now sample read only support csv and parquet")
        return get_dataframe
    
    def execute_pd(self, pipeline, trans_type = 'fit_transform'):
        assert not self.op.children or len(self.op.children) == 0
        _proc = self.get_function_pd(trans_type)
        self.cache = _proc()
    
    def execute_spark(self, pipeline, rdp, trans_type = 'fit_transform'):
        assert not self.op.children or len(self.op.children) == 0
        _proc = self.get_function_spark(rdp, trans_type)
        self.cache = _proc()

AUTOFEOPERATORS.register(DataLoader, "DataLoader")