from .base import BaseOperation
from .featuretools_adaptor import FeaturetoolsOperation

class HaversineOperation(FeaturetoolsOperation):
    def __init__(self, op_base):
        super().__init__(op_base)

    def get_function_pd(self):
        def generate_ft_feature(df):
            for inputs_str, op in self.feature_in_out_map.items():
                inputs = eval(inputs_str)
                op_object = op[1]()
                df[op[0]] = op_object(df[inputs[0]], df[inputs[1]])
            return df
        return generate_ft_feature

class CoordinatesOperation(BaseOperation):        
    def __init__(self, op_base):
        super().__init__(op_base)
        self.points = op_base.config

    def get_function_pd(self):
        def type_infer(df):            
            for p in self.points:
                df = p.get_function()(df)
            return df
        return type_infer