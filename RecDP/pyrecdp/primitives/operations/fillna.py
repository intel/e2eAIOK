from .base import BaseOperation

class FillNaOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self._fillna_feature_map = op_base.config
    
    def get_function_pd(self):
        def fill_na(df):
            df.fillna(self._fillna_feature_map, inplace=True, downcast=False)
            return df
        return fill_na