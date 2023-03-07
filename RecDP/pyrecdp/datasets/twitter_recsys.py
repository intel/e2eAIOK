from .base_api import base_api

class twitter_recsys(base_api):
    def __init__(self):
        super().__init__()
        self.saved_path = self.download_s3("pyrecdp-testdata", "test_twitter_recsys.parquet")

    def to_pandas(self, nrows = None):
        import pandas as pd
        return pd.read_parquet(self.saved_path)
         