from .base_api import base_api

class amazon_product_review(base_api):
    def __init__(self):
        super().__init__()
        name = "amazon_reviews_us_Books.tsv"
        url = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_00.tsv.gz"            
        self.saved_path = self.download_url(name, url, unzip = True)

    def to_pandas(self, nrows = None):
        import pandas as pd
        df = pd.read_table(self.saved_path, on_bad_lines='skip')
        
        # fix train
        df = df.loc[df['star_rating'].apply(lambda x: len(str(x)) <= 3)]
        df['star_rating'] = df['star_rating'].astype(float)
        
        return df
         