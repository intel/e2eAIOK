from .base_api import base_api

class nyc_taxi(base_api):
    def __init__(self):
        super().__init__()
        self.name = "nyc_taxi_fare_cleaned.csv"
        self.url = "https://huggingface.co/datasets/Chendi/NYC_TAXI_FARE_CLEANED/resolve/main/nyc_taxi_fare_cleaned.csv"
        
        self.saved_path = self.download(self.name, self.url)

    def to_pandas(self, nrows = None):
        import pandas as pd
        if nrows:
            return pd.read_csv(self.saved_path, nrows = nrows)
        else:
            return pd.read_csv(self.saved_path)
         