from .base import BaseOperation, AUTOFEOPERATORS
import copy

class CustomOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.custom_op = self.op.config["func_name"]
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self, trans_type = 'fit_transform'):
        return self.custom_op
AUTOFEOPERATORS.register(CustomOperation, "custom_operator")