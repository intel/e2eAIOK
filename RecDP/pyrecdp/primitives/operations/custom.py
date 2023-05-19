from .base import BaseOperation
import copy

class CustomOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.custom_op = self.op.config["func_name"]
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
    
    def get_function_pd(self):
        return self.custom_op