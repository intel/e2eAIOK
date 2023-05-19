from .base import BaseOperation
import copy
from pyrecdp.core.utils import class_name_fix

class FeaturetoolsOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out_map = self.op.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True

    def get_function_pd(self):
        feature_in_out_map = copy.deepcopy(self.feature_in_out_map)
        def generate_ft_feature(df):
            for in_feat_name, ops in feature_in_out_map.items():
                if in_feat_name in df.columns:
                    for op in ops:
                        op_object = class_name_fix(op[1])()
                        df[op[0]] = op_object(df[in_feat_name])
            return df
        return generate_ft_feature
    
    def get_function_spark(self, rdp):
        raise NotImplementedError(f"operations based on featuretools are not support Spark DataFrame yet.")