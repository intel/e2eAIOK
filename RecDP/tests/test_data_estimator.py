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
from pyrecdp.autofe import FeatureWrangler, DataEstimator
#from IPython.display import display


class TestDataEstimatorPandasBased(unittest.TestCase):
        
    def test_frauddetect_train(self):
        from pyrecdp.datasets import ibm_fraud_detect
        train_data = ibm_fraud_detect().to_pandas('test')
        data_pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
        def train_test_splitter(df):
            test_sample = df[df['Year'] == 2018]
            train_sample = df[df['Year'] < 2018]
            return train_sample, test_sample

        config = {
            'model_file': 'test_frauddetect.mdl',
            'metrics': 'auc', 
            'objective': 'binary', 
            'model_name': 'lightgbm',
            'train_test_splitter': train_test_splitter}
        train_pipeline = DataEstimator(method = 'train', data_pipeline = data_pipeline, config = config)
        train_pipeline.fit_transform()
        train_pipeline.export("test_fraud_detect_pipeline.json")
        
    def test_frauddetect_test(self):
        from pyrecdp.datasets import ibm_fraud_detect
        test_data = ibm_fraud_detect().to_pandas('test')
        config = {'dataset': test_data, 'label': 'Is Fraud?'}
        predict_pipeline = DataEstimator(data_pipeline = "test_fraud_detect_pipeline.json", method = 'predict', config = config)
        predict_pipeline.fit_transform()
        
    def test_frauddetect_train_with_json(self):
        from pyrecdp.datasets import ibm_fraud_detect
        test_data = ibm_fraud_detect().to_pandas('test')
        config = {'dataset': test_data, 'label': 'Is Fraud?'}
        train_pipeline = DataEstimator(data_pipeline = "test_fraud_detect_pipeline.json", method = 'train', config = config)
        train_pipeline.fit_transform()
        
    def test_frauddetect_train_with_transformed(self):
        from pyrecdp.datasets import ibm_fraud_detect
        train_data = ibm_fraud_detect().to_pandas('test')
        data_pipeline = FeatureWrangler(dataset=train_data, label="Is Fraud?")
        transformed_data = data_pipeline.fit_transform()
        #display(transformed_data.dtypes)
        def train_test_splitter(df):
            test_sample = df[df['Year'] == 2018]
            train_sample = df[df['Year'] < 2018]
            return train_sample, test_sample

        config = {
            'model_file': 'test_frauddetect.mdl',
            'metrics': 'auc', 
            'objective': 'binary', 
            'model_name': 'lightgbm',
            'train_test_splitter': train_test_splitter}
        train_pipeline = DataEstimator(method = 'train', data_pipeline = data_pipeline, config = config)
        train_pipeline.fit_transform()
        train_pipeline.export("test_fraud_detect_pipeline.json")