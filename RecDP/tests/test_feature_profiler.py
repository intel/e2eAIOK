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
from pyrecdp.autofe import FeatureProfiler
from IPython.display import display


class TestFeatureProfiler(unittest.TestCase):

    def test_nyc_taxi(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureProfiler(dataset=train_data, label="fare_amount")
        display(pipeline.data_stats)
        
    def test_twitter_recsys(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureProfiler(dataset=train_data, label="reply")
        display(pipeline.export())
    
    def test_amazon(self):
        train_data = pd.read_table(f"{pathlib}/tests/data/amazon_reviews_us_Books.tsv", on_bad_lines='skip')
        pipeline = FeatureProfiler(dataset=train_data, label="star_rating")
        display(pipeline.export())
        
    def test_frauddetect(self):
        from pyrecdp.datasets import ibm_fraud_detect
        train_data = ibm_fraud_detect().to_pandas('test')
        pipeline = FeatureProfiler(dataset=train_data, label="Is Fraud?")
        display(pipeline.export())
