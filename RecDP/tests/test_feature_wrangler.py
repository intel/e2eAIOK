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
from pyrecdp.core.utils import Timer
from IPython.display import display


class TestFeatureWranglerPandasBased(unittest.TestCase):

    def test_nyc_taxi(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        display(ret_df)
    
    def test_nyc_taxi_perf(self):
        train_data = pd.read_csv(f"/home/vmagent/app/dataset/nyc_taxi/train.csv")
        with Timer("NYC taxi performance test"):
            pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
            ret_df = pipeline.fit_transform(engine_type = 'pandas')
        display(ret_df)
        
    def test_twitter_recsys(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        display(ret_df)
    
    def test_amazon(self):
        train_data = pd.read_table(f"{pathlib}/tests/data/test_amz.tsv", on_bad_lines='skip')
        pipeline = FeatureWrangler(dataset=train_data, label="star_rating")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        display(ret_df)
        
    def test_frauddetect(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        display(ret_df)
    
    def test_frauddetect_perf(self):
        train_data = pd.read_csv(f"/home/vmagent/app/dataset/fraud_detect/card_transaction.v1.csv")
        with Timer("Fraud detect performance test"):
            pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
            ret_df = pipeline.fit_transform(engine_type = 'pandas')
        display(ret_df)

    def test_ppdt(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/recsys2023_train.parquet")
        #train_data = train_data[:10000]
        pipeline = FeatureWrangler(dataset=train_data, label="is_installed", time_series = 'f_1')
        # pipeline.export(f"{pathlib}/tests/ppdt_pipeline.json")
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        display(ret_df)
    
    def test_recsys2023_perf(self):
        train_data = pd.read_parquet(f"/home/vmagent/app/dataset/recsys2023/recsys2023_train.parquet")
        with Timer("Recsys2023 performance test"):
            pipeline = FeatureWrangler(dataset=train_data, label="is_installed", time_series = 'f_1')
            ret_df = pipeline.fit_transform(engine_type = 'pandas')
        display(ret_df)
        
class TestFeatureWranglerSparkBased(unittest.TestCase):

    def test_nyc_taxi(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        # test with shape
        display(ret_df)
    
    def test_nyc_taxi_perf(self):
        train_data = pd.read_csv(f"/home/vmagent/app/dataset/nyc_taxi/train.csv")
        with Timer("NYC taxi performance test"):
            pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")
            ret_df = pipeline.fit_transform(engine_type = 'spark')
        display(ret_df)
        
    def test_twitter_recsys(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        # test with shape
        display(ret_df)

    def test_amazon(self):
        train_data = pd.read_table(f"{pathlib}/tests/data/test_amz.tsv", on_bad_lines='skip')
        pipeline = FeatureWrangler(dataset=train_data, label="star_rating")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        # test with shape
        display(ret_df)

    def test_frauddetect(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        display(ret_df)

    def test_frauddetect_perf(self):
        train_data = pd.read_csv(f"/home/vmagent/app/dataset/fraud_detect/card_transaction.v1.csv")
        with Timer("Fraud detect performance test"):
            pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
            ret_df = pipeline.fit_transform(engine_type = 'spark')
        display(ret_df)
    
    def test_ppdt(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/recsys2023_train.parquet")
        #train_data = train_data[:10000]
        pipeline = FeatureWrangler(dataset=train_data, label="is_installed", time_series = 'f_1')
        # pipeline.export(f"{pathlib}/tests/ppdt_pipeline.json")
        ret_df = pipeline.fit_transform(engine_type = 'spark')
        display(ret_df)
    
    def test_recsys2023_perf(self):
        train_data = pd.read_parquet(f"/home/vmagent/app/dataset/recsys2023/recsys2023_train.parquet")
        with Timer("Recsys2023 performance test"):
            pipeline = FeatureWrangler(dataset=train_data, label="is_installed", time_series = 'f_1')
            ret_df = pipeline.fit_transform(engine_type = 'spark')
        display(ret_df)