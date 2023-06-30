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

class GroupedCategorifyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    def get_function_pd(self):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        def group_categorify(df, feature_name, grouped_features, dict_path = None):
            encoder = get_encoder_df(dict_path)
            if isinstance(encoder, type(None)):
                k = [i for i in df.columns if i not in grouped_features]
                if len(k) == 0:
                    return ret, None, 0
                k = k[0]
                encoder = df.groupby(by = grouped_features, as_index = False)[k].count().drop(k, axis = 1)
                encoder[feature_name] = encoder.index
                save_encoder_df(encoder, dict_path)
            ret = df.merge(encoder, on = grouped_features, how = 'left')
            return ret
        def categorify(df):
            for feature_out, feature in feature_in_out.items():
                dict_path = None
                df = group_categorify(df, feature_out, feature, dict_path)
            return df
        return categorify