from .base import BaseOperation
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyrecdp.core.utils import *
import copy
from sklearn.preprocessing import LabelEncoder
from pyrecdp.core.parallel_iterator import ParallelIterator
from IPython.display import display

class CategorifyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def label_encode(cls, item):
        feature, df_x, dict_path, feature_out = item
        encoder = LabelEncoder()
        ret = pd.DataFrame()
        ret[feature_out] = encoder.fit_transform(df_x)
        ret.index = df_x.index
        return ret

    def get_function_pd(self):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        def categorify(df):
            df_features = [(col, df[col], dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
            results = ParallelIterator.execute(df_features, CategorifyOperation.label_encode, len(df_features), "Categorify")
            df = df.loc[:, ~df.columns.isin([sdf.columns[0] for sdf in results])]
            df = pd.concat([df] + results, axis=1)
            return df

        return categorify

class GroupCategorifyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    @classmethod
    def group_label_encode(cls, item):
        grouped_features, df_x, dict_path, feature_out = item
        df_x['index'] = df_x.index
        k = 'index'
        # print(grouped_features)
        encoder = df_x.groupby(by = grouped_features, as_index = False)[k].count().drop(k, axis = 1)
        encoder[feature_out] = encoder.index
        #display(encoder)
        ret = df_x.merge(encoder, on = grouped_features, how = 'left')[feature_out]
        return ret
    
    def get_function_pd(self):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        def group_categorify(df):
            df_features = [(group_parts, df[group_parts], dict_path, feature_out) for col, (group_parts, dict_path, feature_out) in feature_in_out.items()]
            results = ParallelIterator.execute(df_features, GroupCategorifyOperation.group_label_encode, len(df_features), "GroupCategorify")
            to_concat_df = pd.concat(results, axis=1)
            to_concat_df.index = df.index
            df = pd.concat([df, to_concat_df], axis=1)
            return df
        return group_categorify