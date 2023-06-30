from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import copy
import numpy as np
from pyrecdp.core.utils import *
from pyrecdp.core.parallel_iterator import ParallelIterator
from IPython.display import display

class OnehotEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
    
    def get_function_pd(self):
        config = copy.deepcopy(self.config)
        def encode(df):
            for feature, keys in config.items():
                selected_columns = [f"{feature}__{key}" for key in keys]
                one_hot_df = pd.get_dummies(df[feature], prefix = f"{feature}_")
                one_hot_df = one_hot_df.loc[:, one_hot_df.columns.isin(selected_columns)]
                df = pd.concat([df, one_hot_df], axis=1)
            return df
        return encode
    
class ListOnehotEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
    
    def get_function_pd(self):
        config = copy.deepcopy(self.config)
        def encode(df):
            from sklearn.preprocessing import MultiLabelBinarizer
            
            for feature, (sep, keys) in config.items():
                splitted_s = df[feature].str.split(sep).apply(lambda x: [i for i in x if i != ""])
                mlb = MultiLabelBinarizer()
                encoded = mlb.fit_transform(splitted_s)
                names = [f"{feature}_{key}" for key in mlb.classes_]
                selected_columns = [f"{feature}_{key}" for key in keys]
                one_hot_df = pd.DataFrame(encoded, columns=names)
                one_hot_df = one_hot_df.loc[:, one_hot_df.columns.isin(selected_columns)]
                df = pd.concat([df, one_hot_df], axis=1)
            return df
        return encode
    

class TargetEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config['feature_in_out']
        self.label = self.op.config['label']
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
    
    def get_function_pd(self):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        label = self.label
        def encode(df):
            df_y = df[label]
            df_features = [(col, df[col], df_y) for col in feature_in_out.keys()]
            te_in_out = zip(df_features, feature_in_out.values())
            parallel_iter = ParallelIterator(te_in_out, target_encode, len(feature_in_out), desc="TargetEncode")
            results = parallel_iter()
            to_concat_df = pd.concat(results, axis=1)
            df = pd.concat([df, to_concat_df], axis=1)
            return df
        return encode

class CountEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
    
    def get_function_pd(self):
        feature_in = copy.deepcopy(self.feature_in)
        def encode(df):
            df_features = [df[col] for col in feature_in]
            ce_items = zip(feature_in, df_features)
            parallel_iter = ParallelIterator(ce_items, count_encode, len(feature_in), desc="CountEncode")
            results = parallel_iter()
            to_concat_df = pd.concat(results, axis=1)
            df = pd.concat([df, to_concat_df], axis=1)
            return df
        return encode