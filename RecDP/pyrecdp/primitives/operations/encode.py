"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from .base import BaseOperation, AUTOFEOPERATORS
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import copy
import numpy as np
from pyrecdp.core.utils import *
from pyrecdp.core.parallel_iterator import ParallelIterator
from IPython.display import display
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from category_encoders import TargetEncoder
from category_encoders.count import CountEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from pyrecdp.encoder import TargetEncoder as SparkTargetEncoder, CountEncoder as SparkCountEncoder
from pyrecdp.data_processor import ModelMerge


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
        one_hot_df = pd.get_dummies(df_x, prefix = f"{feature}_", dtype=int)
        one_hot_df = one_hot_df.loc[:, one_hot_df.columns.isin(selected_columns)]
        return one_hot_df

    def get_function_pd(self, trans_type = 'fit_transform'):
        config = copy.deepcopy(self.config)
        def encode(df):
            df_features = [(col, df[col], keys) for col, keys in config.items()]
            results = ParallelIterator.execute(df_features, OnehotEncodeOperation.onehot_encode, len(df_features), "OnehotEncode")
            df = pd.concat([df] + results, axis=1)
            return df
        return encode
AUTOFEOPERATORS.register(OnehotEncodeOperation, "onehot_encode")
    
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
    
    def get_function_pd(self, trans_type = 'fit_transform'):
        config = copy.deepcopy(self.config)
        def encode(df):
            df_features = [(col, df[col], sep, keys) for col, (sep, keys) in config.items()]
            results = ParallelIterator.execute(df_features, ListOnehotEncodeOperation.multi_label_encode, len(df_features), "ListOnehotEncode")
            df = pd.concat([df] + results, axis=1)
            return df
        return encode
AUTOFEOPERATORS.register(ListOnehotEncodeOperation, "list_onehot_encode")

class TargetEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config['feature_in_out']
        self.label = self.op.config['label']
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def target_encode(cls, item):
        feature, df_x, df_y, dict_path, feature_out = item
        if is_unique(df_x):
            df_encoded = pd.DataFrame()
            df_encoded[feature_out] = [np.nan]*len(df_x)
            df_encoded.index = df_x.index
        else:
            encoder = TargetEncoder(cols=[feature], min_samples_leaf=20, smoothing=10)
            df_encoded = encoder.fit_transform(df_x, df_y).rename(columns={feature: feature_out})
            save_encoder(encoder, dict_path)
        return df_encoded

    @classmethod
    def target_encode_transform(cls, item):
        feature, df_x, df_y, dict_path, feature_out = item
        encoder = get_encoder(dict_path)
        df_encoded = encoder.transform(df_x).rename(columns={feature: feature_out})
        return df_encoded

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        label = self.label

        if trans_type == 'fit_transform':
            def encode(df):
                df_y = df[label]
                df_features = [(col, df[col], df_y, dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, TargetEncodeOperation.target_encode, len(feature_in_out), "TargetEncode")
                to_concat_df = pd.concat(results, axis=1)
                df = pd.concat([df, to_concat_df], axis=1)
                return df
        elif trans_type == 'transform':
            def encode(df):
                df_features = [(col, df[col], None, dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, TargetEncodeOperation.target_encode_transform, len(feature_in_out), "TargetEncode")
                to_concat_df = pd.concat(results, axis=1)
                df = pd.concat([df, to_concat_df], axis=1)
                return df
        return encode

    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        label = self.label
        
        def encode(df):
            te_dfs = []
            for f_in, f_out in feature_in_out.items():
                encoder = SparkTargetEncoder(rdp, f_in, [label], [f_out], f"{f_in}_TE")
                _, te_df = encoder.transform(df)
                te_dfs.append({'col_name': f_in, 'dict': te_df})
            op_merge_TE = ModelMerge(te_dfs)
            rdp.reset_ops([op_merge_TE])
            df = rdp.apply(df)
            return df
        return encode
AUTOFEOPERATORS.register(TargetEncodeOperation, "target_encode")

class CountEncodeOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def count_encode(cls, item):
        dict_path = None
        feature, df_x, dict_path, feature_out = item
        if is_unique(df_x):
            df_encoded = pd.DataFrame()
            df_encoded[feature_out] = [np.nan]*len(df_x)
            df_encoded.index = df_x.index
        else:
            encoder = CountEncoder(cols=[feature], handle_unknown='return_nan')
            df_encoded = encoder.fit_transform(df_x).rename(columns={feature: feature_out})
            save_encoder(encoder, dict_path)
        return df_encoded

    @classmethod
    def count_encode_transform(cls, item):
        from pyrecdp.core.utils import fillna_with_series
        dict_path = None
        feature, df_x, dict_path, feature_out = item
        encoder = get_encoder(dict_path)
        df_encoded = encoder.transform(df_x).rename(columns={feature: feature_out})
        # handle unknown

        new_encoder = CountEncoder(cols=[feature], handle_unknown='return_nan')
        df_encoded_2 = new_encoder.fit_transform(df_x).rename(columns={feature: feature_out})

        df_encoded[feature_out] = fillna_with_series(df_encoded[feature_out], df_encoded_2[feature_out])

        # print debug
        # df_encoded[feature] = df_x
        # display(df_encoded[df_encoded[feature].isin([25855, 30061])])
        return df_encoded

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in_out = copy.deepcopy(self.feature_in_out)

        if trans_type == 'fit_transform':
            def encode(df):
                df_features = [(col, df[col], dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, CountEncodeOperation.count_encode, len(df_features), "CountEncode")
                to_concat_df = pd.concat(results, axis=1)
                df = pd.concat([df, to_concat_df], axis=1)
                return df
        elif trans_type == 'transform':
            def encode(df):
                df_features = [(col, df[col], dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, CountEncodeOperation.count_encode_transform, len(df_features), "CountEncode")
                to_concat_df = pd.concat(results, axis=1)
                df = pd.concat([df, to_concat_df], axis=1)
                return df
        return encode

    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        feature_in = copy.deepcopy(self.feature_in)
        def encode(df):
            ce_dfs = []
            for col in feature_in:
                encoder = SparkCountEncoder(rdp, col, [col], [f"{col}_CE"], f"{col}_CE", False)
                ce_df = encoder.transform(df)
                ce_dfs.append({'col_name': col, 'dict': ce_df})
            op_merge_CE = ModelMerge(ce_dfs)
            rdp.reset_ops([op_merge_CE])
            df = rdp.apply(df)
            return df
        return encode
AUTOFEOPERATORS.register(CountEncodeOperation, "count_encode")