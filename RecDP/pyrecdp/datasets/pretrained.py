from .base_api import base_api

class pretrained(base_api):
    def __init__(self):
        super().__init__()
   
    def download(self, model_name):
        if model_name == "nyc_taxi_fare":
            name = "lightgbm_regression_nyc_taxi_fare_amount.mdl"
            url = f"https://pyrecdp-testdata.s3.us-west-2.amazonaws.com/{name}"            
            return self.download_url(name, url)
         