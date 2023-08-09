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
from pyrecdp.autofe import FeatureWrangler, BasePipeline

cur_dir = str(Path(__file__).parent.resolve())

class TestPipielineJson(unittest.TestCase):

    def test_import_nyc(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")

        # export pipeline to json
        pipeline.export(f"{cur_dir}/nyc_taxi_fare_pipeline.json")

        # create a brand-new pipeline and import from json
        new_pipeline = BasePipeline(dataset=train_data, label="fare_amount")
        new_pipeline.import_from_json(f"{cur_dir}/nyc_taxi_fare_pipeline.json")
        new_pipeline.fit_transform()
        
    def test_import_amazon(self):
        train_data = pd.read_table(f"{pathlib}/tests/data/test_amz.tsv", on_bad_lines='skip')
        pipeline = FeatureWrangler(dataset=train_data, label="star_rating")
        
        # export pipeline to json
        pipeline.export(f"{cur_dir}/amazon_pipeline.json")

        # create a brand-new pipeline and import from json
        new_pipeline = BasePipeline(dataset=train_data, label="star_rating")
        new_pipeline.import_from_json(f"{cur_dir}/amazon_pipeline.json")
        new_pipeline.fit_transform()
        
    def test_import_twitter(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        
        # export pipeline to json
        pipeline.export(f"{cur_dir}/twitter_pipeline.json")

        # create a brand-new pipeline and import from json
        new_pipeline = BasePipeline(dataset=train_data, label="reply")
        new_pipeline.import_from_json(f"{cur_dir}/twitter_pipeline.json")
        new_pipeline.fit_transform()

    def test_import_nyc_execute_spark(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_nyc_taxi_fare.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="fare_amount")

        # export pipeline to json
        pipeline.export(f"{cur_dir}/nyc_taxi_fare_pipeline.json")

        # create a brand-new pipeline and import from json
        new_pipeline = BasePipeline(dataset=train_data, label="fare_amount")
        new_pipeline.import_from_json(f"{cur_dir}/nyc_taxi_fare_pipeline.json")
        new_pipeline.fit_transform('spark')
        

    def test_import_amazon_execute_spark(self):
        train_data = pd.read_table(f"{pathlib}/tests/data/test_amz.tsv", on_bad_lines='skip')
        pipeline = FeatureWrangler(dataset=train_data, label="star_rating")
        
        # export pipeline to json
        pipeline.export(f"{cur_dir}/amazon_pipeline.json")

        # create a brand-new pipeline and import from json
        new_pipeline = BasePipeline(dataset=train_data, label="star_rating")
        new_pipeline.import_from_json(f"{cur_dir}/amazon_pipeline.json")
        new_pipeline.fit_transform('spark')
        
    def test_import_twitter_execute_spark(self):
        train_data = pd.read_parquet(f"{pathlib}/tests/data/test_twitter_recsys.parquet")
        pipeline = FeatureWrangler(dataset=train_data, label="reply")
        
        # export pipeline to json
        pipeline.export(f"{cur_dir}/twitter_pipeline.json")

        # create a brand-new pipeline and import from json
        new_pipeline = BasePipeline(dataset=train_data, label="reply")
        new_pipeline.import_from_json(f"{cur_dir}/twitter_pipeline.json")
        new_pipeline.fit_transform('spark')
