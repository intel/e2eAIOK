from .base_api import base_api

class nyc_taxi(base_api):
    def __init__(self, scale = 'test'):
        super().__init__()
        self.scale = scale
        if scale == 'test':            
            self.saved_path = self.download_s3("pyrecdp-testdata", "test_nyc_taxi_fare.parquet")
        elif scale == 'test_large':
            self.saved_path = self.download_s3("pyrecdp-testdata", "nyc_taxi_fare_1M.csv")
        elif scale == 'full':
            self.name = "nyc_taxi_fare_cleaned.csv"
            self.url = "https://huggingface.co/datasets/Chendi/NYC_TAXI_FARE_CLEANED/resolve/main/nyc_taxi_fare_cleaned.csv"
            
            self.saved_path = self.download_url(self.name, self.url)

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
         