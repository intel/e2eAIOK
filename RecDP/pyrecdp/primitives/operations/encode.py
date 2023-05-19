from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import copy
import numpy as np
from pyrecdp.core.utils import *
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
                splitted_s = df[feature].str.split(sep).apply(lambda x: [i for i in x if i is not ""])
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
        self.feature_in = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self):
        from category_encoders import TargetEncoder
        feature_in = copy.deepcopy(self.feature_in)
        def encode(df):
            added_data = {}
            for feature, (dict_path, target_label) in feature_in.items():
                encoder = TargetEncoder(cols=[feature], min_samples_leaf=20, smoothing=10)
                encoder = get_encoder_np(encoder, dict_path)
                added_data[f"{feature}_TE"] = encoder.fit_transform(df[feature], df[target_label])[feature]
                save_encoder_np(encoder, dict_path)
            to_concat_df = pd.DataFrame.from_dict(data = added_data)
            df = pd.concat([df, to_concat_df], axis = 1)
            return df
        return encode
    
class CountEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self):
        from category_encoders.count import CountEncoder
        feature_in = copy.deepcopy(self.feature_in)
        def encode(df):
            added_data = {}
            for feature in feature_in:
                dict_path = None
                encoder = CountEncoder(cols=[feature])
                encoder = get_encoder_np(encoder, dict_path)
                added_data[f"{feature}_CE"] = encoder.fit_transform(df[feature])[feature]
                save_encoder_np(encoder, dict_path)
            to_concat_df = pd.DataFrame.from_dict(data = added_data)
            df = pd.concat([df, to_concat_df], axis = 1)
            return df
        return encode