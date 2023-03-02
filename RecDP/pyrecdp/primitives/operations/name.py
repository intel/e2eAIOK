from .base import BaseOperation

class RenameOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.renamed = op_base.config
        
    def get_function_pd(self):
        def rename(df):
            return df.rename(columns = self.renamed)
        return rename

    def get_function_spark(self, rdp):
        def rename(df):
            for src, dst in self.renamed.items():
                df = df.withColumnRenamed(src, dst)
            
            return df
        return rename