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
from pyrecdp.autofe import FeatureWrangler, FeatureEstimator

import warnings
warnings.filterwarnings("ignore")

#from IPython.display import display


class TestFeatureEstimatorPandasBased(unittest.TestCase):
        
    def test_frauddetect(self):
        from pyrecdp.datasets import ibm_fraud_detect
        train_data = ibm_fraud_detect().to_pandas('test')
        data_pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")

        config = {
            'model_file': 'test_frauddetect.mdl',
            'objective': 'binary', 
            'model_name': 'lightgbm'}
        train_pipeline = FeatureEstimator(data_pipeline = data_pipeline, config = config)
        ret = train_pipeline.fit_transform()
        print(train_pipeline.get_feature_importance())
        
    def test_frauddetect_with_valid(self):
        from pyrecdp.datasets import ibm_fraud_detect
        train_data = ibm_fraud_detect().to_pandas('test')
        data_pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
        def train_test_splitter(df):
            test_sample = df[df['Year'] == 2018]
            train_sample = df[df['Year'] < 2018]
            return train_sample, test_sample

        config = {
            'model_file': 'test_frauddetect.mdl',
            'objective': 'binary', 
            'model_name': 'lightgbm',
            'train_test_splitter': train_test_splitter}
        train_pipeline = FeatureEstimator(data_pipeline = data_pipeline, config = config)
        ret = train_pipeline.fit_transform()
        print(train_pipeline.get_feature_importance())