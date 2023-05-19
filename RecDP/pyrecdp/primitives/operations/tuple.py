from .base import BaseOperation

class TupleOperation(BaseOperation):        
    def __init__(self, op_base):
        super().__init__(op_base)
        self.support_spark_dataframe = False
        self.support_spark_rdd = True
        self.feature_in = self.op.config['src']
        self.feature_out = self.op.config['dst']

    def get_function_pd(self):
        feature_in = self.feature_in.copy()
        feature_out = self.feature_out
        def process(df):
            df[feature_out] = df[feature_in].apply(tuple, axis=1)
            return df
        return process

    def get_function_spark(self, rdp):
        raise NotImplementedError(f"CoordinatesOperation spark dataframe is not supported yet.")