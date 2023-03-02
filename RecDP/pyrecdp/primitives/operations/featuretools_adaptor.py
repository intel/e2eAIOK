from .base import BaseOperation

class FeaturetoolsOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.feature_in_out_map = op_base.config

    def get_function_pd(self):
        def generate_ft_feature(df):
            for in_feat_name, ops in self.feature_in_out_map.items():
                for op in ops:
                    op_object = op[1]()
                    df[op[0]] = op_object(df[in_feat_name])
            return df
        return generate_ft_feature