from .base import BaseOperation
from .featuretools_adaptor import FeaturetoolsOperation
import copy
from pyrecdp.core.utils import class_name_fix

class HaversineOperation(FeaturetoolsOperation):
    def __init__(self, op_base):
        super().__init__(op_base)

    def get_function_pd(self):
        feature_in_out_map = copy.deepcopy(self.feature_in_out_map)
        def generate_ft_feature(df):
            for inputs_str, op in feature_in_out_map.items():
                inputs = eval(inputs_str)
                op_object = class_name_fix(op[1])()
                df[op[0]] = op_object(df[inputs[0]], df[inputs[1]])
            return df
        return generate_ft_feature

