from .base import BaseOperation
from pyrecdp.core import SeriesSchema
import pandas as pd
import copy

class TypeInferOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.astype_feature_map = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True

    def get_function_pd(self):
        astype_feature_map = copy.deepcopy(self.astype_feature_map)
        def type_infer(df):
            for feature in astype_feature_map:
                feature_name, dest_type_list = feature[0], feature[1]
                # if 'is_datetime' in dest_type_list:
                #     df[feature_name] = pd.to_datetime(df[feature_name], errors='coerce', infer_datetime_format=True)
                if 'is_numeric' in dest_type_list:
                    df[feature_name] = pd.to_numeric(df[feature_name], errors='coerce')
            return df
        return type_infer
    
    def get_function_spark(self, rdp):
        raise NotImplementedError(f"TypeInferOperation spark dataframe is not supported yet.")