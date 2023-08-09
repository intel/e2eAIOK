from .base_api import base_api

class nyc_taxi(base_api):
    def __init__(self, scale = 'full'):
        super().__init__()
        self.scale = scale
        if scale == 'test':
            name = "test_nyc_taxi_fare.parquet"
            url = f"https://pyrecdp-testdata.s3.us-west-2.amazonaws.com/{name}"  
        elif scale == 'test_large':
            name = "nyc_taxi_fare_1M.csv"
            url = f"https://pyrecdp-testdata.s3.us-west-2.amazonaws.com/{name}"
        elif scale == 'full':
            name = "nyc_taxi_fare_cleaned.csv"
            url = "https://huggingface.co/datasets/Chendi/NYC_TAXI_FARE_CLEANED/resolve/main/nyc_taxi_fare_cleaned.csv"
                  
        self.saved_path = self.download_url(name, url)

    def to_pandas(self, nrows = None):
        import pandas as pd
        if self.scale == 'test':            
            return pd.read_parquet(self.saved_path)
        elif self.scale == 'test_large':
            return pd.read_csv(self.saved_path)
        elif self.scale == 'full':            
            if nrows:
                return pd.read_csv(self.saved_path, nrows = nrows)
            else:
                return pd.read_csv(self.saved_path)
         