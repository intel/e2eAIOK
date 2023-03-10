from .base_api import base_api

class twitter_recsys(base_api):
    def __init__(self):
        super().__init__()
        name = "test_twitter_recsys.parquet"
        url = f"https://pyrecdp-testdata.s3.us-west-2.amazonaws.com/{name}"            
        self.saved_path = self.download_url(name, url)

    def to_pandas(self, nrows = None):
        import pandas as pd
        return pd.read_parquet(self.saved_path)
         