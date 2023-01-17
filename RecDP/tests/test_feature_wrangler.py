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


class TestFeatureWranglerPandasBased(unittest.TestCase):

    def test_nyc_taxi(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        self.assertEqual(ret_df.shape[0], 10000)
        self.assertTrue(ret_df.shape[1] >= 15)
        
    def test_twitter_recsys(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        self.assertEqual(ret_df.shape[0], 10000)
        self.assertTrue(ret_df.shape[1] >= 41)
        
class TestFeatureWranglerSparkBased(unittest.TestCase):

    def test_nyc_taxi(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        # test with shape
        self.assertEqual(ret_df.shape[0], 10000)
        self.assertTrue(ret_df.shape[1] >= 15)
        
    def test_twitter_recsys(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        self.assertEqual(ret_df.shape[0], 10000)
        self.assertTrue(ret_df.shape[1] >= 41)