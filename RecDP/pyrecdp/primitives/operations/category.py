from .base import BaseOperation, AUTOFEOPERATORS
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
        save_encoder(encoder, dict_path)
        ret.index = df_x.index
        #display(encoder.classes_)
        return ret

    @classmethod
    def label_encode_transform(cls, item):
        feature, df_x, dict_path, feature_out = item
        encoder = get_encoder(dict_path)
        # combine encoder with new data
        encoder_class_list = encoder.classes_.tolist()
        encoder_new = LabelEncoder()
        encoder_new.fit(df_x)
        combined_classes = encoder_class_list + [i for i in encoder_new.classes_ if i not in encoder_class_list]
        encoder.classes_ = np.array(combined_classes)
        # start to transform
        ret = pd.DataFrame()
        ret[feature_out] = encoder.transform(df_x)
        ret.index = df_x.index
        #display(encoder.classes_)
        return ret

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        if trans_type == 'fit_transform':
            def categorify(df):
                df_features = [(col, df[col], dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, CategorifyOperation.label_encode, len(df_features), "Categorify")
                df = df.loc[:, ~df.columns.isin([sdf.columns[0] for sdf in results])]
                df = pd.concat([df] + results, axis=1)
                return df
        elif trans_type == 'transform':
            def categorify(df):
                df_features = [(col, df[col], dict_path, feature_out) for col, (dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, CategorifyOperation.label_encode_transform, len(df_features), "Categorify")
                df = df.loc[:, ~df.columns.isin([sdf.columns[0] for sdf in results])]
                df = pd.concat([df] + results, axis=1)
                return df

        return categorify
AUTOFEOPERATORS.register(CategorifyOperation, "categorify")

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
        save_encoder(encoder, dict_path)
        #display(encoder)
        ret = df_x.merge(encoder, on = grouped_features, how = 'left')[feature_out]
        return ret

    @classmethod
    def group_label_encode_transform(cls, item):
        grouped_features, df_x, dict_path, feature_out = item
        encoder_df = get_encoder(dict_path)
        # convert df to dict
        key_feats = [i for i in encoder_df.columns if i != feature_out]
        encoder_df['key'] = encoder_df[key_feats].apply(lambda x: str(list(x)), axis=1)
        encoder_dict = dict(zip(encoder_df['key'].to_list(), encoder_df[feature_out].to_list()))
        max_id = max(list(encoder_dict.values())) + 1

        # get sub df
        sub_df = df_x[grouped_features]
        sub_df['key'] = sub_df[key_feats].apply(lambda x: str(list(x)), axis=1)

        cate_id_list = []
        for key_id in sub_df['key'].to_list():
            if key_id not in encoder_dict:
                encoder_dict[key_id] = max_id
                max_id += 1
            cate_id_list.append(encoder_dict[key_id])

        #display(encoder_dict)
        return pd.Series(cate_id_list, name = feature_out, index = sub_df.index)

    def get_function_pd(self, trans_type = 'fit_transform'):
        feature_in_out = copy.deepcopy(self.feature_in_out)
        if trans_type == 'fit_transform':
            def group_categorify(df):
                df_features = [(group_parts, df[group_parts], dict_path, feature_out) for col, (group_parts, dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, GroupCategorifyOperation.group_label_encode, len(df_features), "GroupCategorify")
                to_concat_df = pd.concat(results, axis=1)
                to_concat_df.index = df.index
                df = pd.concat([df, to_concat_df], axis=1)
                return df
        if trans_type == 'transform':
            def group_categorify(df):
                df_features = [(group_parts, df[group_parts], dict_path, feature_out) for col, (group_parts, dict_path, feature_out) in feature_in_out.items()]
                results = ParallelIterator.execute(df_features, GroupCategorifyOperation.group_label_encode_transform, len(df_features), "GroupCategorify")
                to_concat_df = pd.concat(results, axis=1)
                to_concat_df.index = df.index
                df = pd.concat([df, to_concat_df], axis=1)
                return df
        return group_categorify
AUTOFEOPERATORS.register(GroupCategorifyOperation, "group_categorify")