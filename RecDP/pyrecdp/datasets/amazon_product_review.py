from .base_api import base_api

class amazon_product_review(base_api):
    def __init__(self):
        super().__init__()
        self.saved_path = self.download_s3("pyrecdp-testdata", "amazon_reviews_us_Books.tsv")

    def to_pandas(self, nrows = None):
        import pandas as pd
        return pd.read_table(self.saved_path, on_bad_lines='skip')
         