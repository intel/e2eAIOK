from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import copy
import numpy as np
from pyrecdp.core.utils import *
from pyrecdp.core.parallel_iterator import ParallelIterator
from IPython.display import display
from category_encoders import TargetEncoder
from category_encoders.count import CountEncoder
from sklearn.preprocessing import MultiLabelBinarizer

class OnehotEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def onehot_encode(cls, item):
        feature, df_x, keys = item
        selected_columns = [f"{feature}__{key}" for key in keys]
        one_hot_df = pd.get_dummies(df_x, prefix = f"{feature}_")
        one_hot_df = one_hot_df.loc[:, one_hot_df.columns.isin(selected_columns)]
        return one_hot_df

    def get_function_pd(self):
        config = copy.deepcopy(self.config)
        def encode(df):
            df_features = [(col, df[col], keys) for col, keys in config.items()]
            results = ParallelIterator.execute(df_features, OnehotEncodeOperation.onehot_encode, len(df_features), "OnehotEncode")
            df = pd.concat([df] + results, axis=1)
            return df
        return encode
    
class ListOnehotEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def multi_label_encode(cls, item):
        feature, df_x, sep, keys = item
        splitted_s = df_x.str.split(sep).apply(lambda x: [i for i in x if i != ""])
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(splitted_s)
        names = [f"{feature}_{key}" for key in mlb.classes_]
        selected_columns = [f"{feature}_{key}" for key in keys]
        one_hot_df = pd.DataFrame(encoded, columns=names)
        one_hot_df = one_hot_df.loc[:, one_hot_df.columns.isin(selected_columns)]
        return one_hot_df
    
    def get_function_pd(self):
        config = copy.deepcopy(self.config)
        def encode(df):
            df_features = [(col, df[col], sep, keys) for col, (sep, keys) in config.items()]
            results = ParallelIterator.execute(df_features, ListOnehotEncodeOperation.multi_label_encode, len(df_features), "ListOnehotEncode")
            df = pd.concat([df] + results, axis=1)
            return df
        return encode

class TargetEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config['feature_in_out']
        self.label = self.op.config['label']
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def target_encode(cls, item):
        (feature, df_x, df_y), feature_out = item
        if is_unique(df_x):
            df_encoded = pd.DataFrame()
            df_encoded[feature_out] = [np.nan]*len(df_x)
            df_encoded.index = df_x.index
        else:
            encoder = TargetEncoder(cols=[feature], min_samples_leaf=20, smoothing=10)
            df_encoded = encoder.fit_transform(df_x, df_y).rename(columns={feature: feature_out})
        return df_encoded

    def get_function_pd(self):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        label = self.label

        def encode(df):
            df_y = df[label]
            df_features = [(col, df[col], df_y) for col in feature_in_out.keys()]
            te_in_out = zip(df_features, feature_in_out.values())
            results = ParallelIterator.execute(te_in_out, TargetEncodeOperation.target_encode, len(feature_in_out), "TargetEncode")
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

    @classmethod
    def count_encode(cls, item):
        dict_path = None
        feature, df_x = item
        if is_unique(df_x):
            df_encoded = pd.DataFrame()
            df_encoded[f"{feature}_CE"] = [np.nan]*len(df_x)
            df_encoded.index = df_x.index
        else:
            encoder = CountEncoder(cols=[feature])
            encoder = get_encoder_np(encoder, dict_path)
            df_encoded = encoder.fit_transform(df_x).rename(columns={feature: f"{feature}_CE"})
            save_encoder_np(encoder, dict_path)
        return df_encoded

    def get_function_pd(self):
        feature_in = copy.deepcopy(self.feature_in)

        def encode(df):
            df_features = [(col, df[col]) for col in feature_in]
            results = ParallelIterator.execute(df_features, CountEncodeOperation.count_encode, len(df_features), "CountEncode")
            to_concat_df = pd.concat(results, axis=1)
            df = pd.concat([df, to_concat_df], axis=1)
            return df
        return encode