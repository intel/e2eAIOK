from .base import BaseOperation
import copy

class FillNaOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self._fillna_feature_map = self.op.config
        self.support_spark_dataframe = True
        self.support_spark_rdd = True
    
    def get_function_pd(self):
        _fillna_feature_map = copy.deepcopy(self._fillna_feature_map)
        def fill_na(df):
            df.fillna(_fillna_feature_map, inplace=True, downcast=False)
            return df
        return fill_na

    def get_function_spark(self, rdp):
        def fill_na(df):
            return df.na.fill(self._fillna_feature_map)
        return fill_na