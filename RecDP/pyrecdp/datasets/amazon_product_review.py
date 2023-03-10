from .base_api import base_api

class amazon_product_review(base_api):
    def __init__(self):
        super().__init__()
        name = "amazon_reviews_us_Books.tsv"
        url = f"https://pyrecdp-testdata.s3.us-west-2.amazonaws.com/{name}"            
        self.saved_path = self.download_url(name, url)

    def to_pandas(self, nrows = None):
        import pandas as pd
        return pd.read_table(self.saved_path, on_bad_lines='skip')
         