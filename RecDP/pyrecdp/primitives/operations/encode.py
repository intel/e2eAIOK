from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import copy
import numpy as np
from IPython.display import display

class OnehotEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = op_base.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
    
    def get_function_pd(self):
        config = copy.deepcopy(self.config)
        def encode(df):
            for feature, keys in config.items():
                selected_columns = [f"{feature}__{key}" for key in keys]
                one_hot_df = pd.get_dummies(df[feature], prefix = f"{feature}_")
                one_hot_df = one_hot_df.loc[:, one_hot_df.columns.isin(selected_columns)]
                #there is possibility designed keys are not included or new keys added
                # ret = {}
                # len_df = one_hot_df.shape[0]
                # for col_name in selected_columns:
                #     if col_name in one_hot_df.columns:
                #         ret[col_name] = one_hot_df[col_name].to_numpy()
                #     else:
                #         ret[col_name] = np.zeros(len_df, dtype=np.uint8)
                # one_hot_df = pd.DataFrame(ret, dtype=np.uint8, index=df.index)
                df = pd.concat([df, one_hot_df], axis=1)
            return df
        return encode
    
class ListOnehotEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = op_base.config
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
                # ret = {}
                # len_df = one_hot_df.shape[0]
                # for col_name in selected_columns:
                #     if col_name in one_hot_df.columns:
                #         ret[col_name] = one_hot_df[col_name].to_numpy()
                #     else:
                #         ret[col_name] = np.zeros(len_df, dtype=np.uint8)
                # one_hot_df = pd.DataFrame(ret, dtype=np.uint8, index=df.index)
                df = pd.concat([df, one_hot_df], axis=1)
            return df
        return encode
    

class TargetEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in = op_base.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self):
        feature_in = copy.deepcopy(self.feature_in)
        def encode(df):
            for feature in feature_in:
                codes, uniques = pd.factorize(df[feature])
                df[f"{feature}__idx"] = pd.Series(codes, df[feature].index)
            return df
        return encode