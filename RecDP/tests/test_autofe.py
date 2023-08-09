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
from pyrecdp.autofe import AutoFE
from IPython.display import display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


class TestFE(unittest.TestCase):

    def test_nyc_taxi_pandas(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = AutoFE(dataset=train_data, label="fare_amount")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        #pipeline.feature_importance()
        
    def test_nyc_taxi_spark(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = AutoFE(dataset=train_data, label="fare_amount")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        # test with shape
        #pipeline.feature_importance()
        
    def test_fraud_detect_pandas(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")

        data = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")
        train_data = data[data['Year'] < 2018].reset_index(drop=True)
        valid_data = data[data['Year'] == 2018].reset_index(drop=True)
        test_data = data[data['Year'] > 2018].reset_index(drop=True)
        target_label = 'Is Fraud?'
        
        pipeline = AutoFE(dataset=train_data, label=target_label, time_series = ['Day', 'Year'])
        transformed_train_df = pipeline.fit_transform()

        valid_pipeline = AutoFE.clone_pipeline(pipeline, valid_data)
        transformed_valid_df = valid_pipeline.transform()

        test_pipeline = AutoFE.clone_pipeline(pipeline, test_data)
        transformed_test_df = test_pipeline.transform()
        
        display(transformed_train_df)
        display(transformed_valid_df)
        display(transformed_test_df)
        
        
    def test_fraud_detect_spark(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")
        pipeline = AutoFE(dataset=train_data, label="Is Fraud?")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        # test with shape
        #pipeline.feature_importance()
        
    # def test_twitter_recsys(self):
    #     train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
    #     pipeline = FeatureWrangler(dataset=train_data, label="reply")
    #     ret_df = pipeline.fit_transform(engine_type = 'pandas')
    #     # test with shape
    #     self.assertEqual(ret_df.shape[0], 10000)
    #     self.assertTrue(ret_df.shape[1] >= 31)
    
    # def test_amazon(self):
    #     train_data = pd.read_table(f"{pathlib}/tests/data/amazon_reviews_us_Books.tsv", on_bad_lines='skip')
    #     pipeline = FeatureWrangler(dataset=train_data, label="star_rating")
    #     ret_df = pipeline.fit_transform(engine_type = 'pandas')
    #     # test with shape
    #     self.assertEqual(ret_df.shape[0], 9667)
    #     self.assertTrue(ret_df.shape[1] >= 16)
        
    # def test_frauddetect(self):
    #     from pyrecdp.datasets import ibm_fraud_detect
    #     train_data = ibm_fraud_detect().to_pandas('test')
    #     pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
    #     ret_df = pipeline.fit_transform(engine_type = 'pandas')
    #     display(ret_df)
        
