from .base_api import base_api

class ibm_fraud_detect(base_api):
    def __init__(self, scale = 'full'):
        super().__init__()
        if scale == 'test':
            raise NotImplementedError("ibm_fraud_detect test dataset is not created yet")
        else:
            url = "https://huggingface.co/datasets/Chendi/ibm_transactions/resolve/main/transactions.tgz"         
            self.saved_path = self.download_url("card_transaction.v1.csv", url, unzip = True)

    def to_pandas(self, scale = 'full'):
        import pandas as pd
        if scale == 'test':
            return pd.read_csv(self.saved_path, nrows = 100000)
        return pd.read_csv(self.saved_path)
         