import unittest
import sys
import pandas as pd
from pathlib import Path
pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)
from pyrecdp.autofe import FeatureWrangler


class TestPipielineJson(unittest.TestCase):

    def test_export_nyc(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
        pipeline.export()

    def test_export_amazon(self):
        train_data = pd.read_table(f"{pathlib}/tests/data/amazon_reviews_us_Books.tsv", on_bad_lines='skip')
        pipeline = FeatureWrangler(dataset=train_data, label="star_rating")
        pipeline.export()
        
    def test_export_twitter(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        pipeline.export()

