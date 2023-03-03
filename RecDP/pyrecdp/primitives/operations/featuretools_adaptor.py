from .base import BaseOperation
import copy

class FeaturetoolsOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out_map = op_base.config
        self.support_spark_dataframe = False
        self.support_spark_rdd = True

    def get_function_pd(self):
        feature_in_out_map = copy.deepcopy(self.feature_in_out_map)
        def generate_ft_feature(df):
            for in_feat_name, ops in feature_in_out_map.items():
                for op in ops:
                    op_object = op[1]()
                    df[op[0]] = op_object(df[in_feat_name])
            return df
        return generate_ft_feature
    
    def get_function_spark(self, rdp):
        raise NotImplementedError(f"operations based on featuretools are not support Spark DataFrame yet.")