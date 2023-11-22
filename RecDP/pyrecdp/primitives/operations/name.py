from .base import BaseOperation, AUTOFEOPERATORS
import copy

class RenameOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.renamed = self.op.config
        self.support_spark_dataframe = True
        self.support_spark_rdd = True
        self.fast_without_dpp = True
        
    def get_function_pd(self, trans_type = 'fit_transform'):
        renamed = copy.deepcopy(self.renamed)
        def rename(df):
            return df.rename(columns = renamed)
        return rename

    def get_function_spark(self, rdp, trans_type = 'fit_transform'):
        def rename(df):
            for src, dst in self.renamed.items():
                df = df.withColumnRenamed(src, dst)
            
            return df
        return rename
AUTOFEOPERATORS.register(RenameOperation, "rename")