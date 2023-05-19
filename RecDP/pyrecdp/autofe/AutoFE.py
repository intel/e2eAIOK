import logging
from pyrecdp.core.utils import Timer, infer_problem_type
from pyrecdp.core.dataframe import DataFrameAPI

from pyrecdp.autofe import FeatureWrangler, FeatureProfiler, RelationalBuilder, FeatureEstimator

import os

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class AutoFE():
    def __init__(self, dataset, label, *args, **kwargs):
        self.label = label
        self.auto_pipeline = {'relational': None, 'profiler': None, 'wrangler': None, 'estimator': None}
        if isinstance(dataset, dict):
            self.auto_pipeline['relational'] = RelationalBuilder(dataset=dataset, label=label)
        else:
            print("AutoFE started to profile data")
            self.auto_pipeline['profiler'] = FeatureProfiler(dataset=dataset, label=label)
            print("AutoFE started to create data pipeline")
            self.auto_pipeline['wrangler'] = FeatureWrangler(dataset=dataset, label=label)
            pipeline = self.auto_pipeline['wrangler']
            config = {
                'model_file': 'autofe_lightgbm.mdl',
                'objective': infer_problem_type(pipeline.dataset[pipeline.main_table][label]),
                'model_name': 'lightgbm'}
            self.auto_pipeline['estimator'] = FeatureEstimator(data_pipeline = pipeline, config = config)

    def fit_transform(self, engine_type = 'pandas', no_cache = False, *args, **kwargs):
        print("AutoFE started to execute data")
        ret_df = None
        if self.auto_pipeline['relational']:
            pipeline = self.auto_pipeline['relational']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)
            print("AutoFE started to profile data")

        if self.auto_pipeline['estimator']:
            pipeline = self.auto_pipeline['estimator']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)
        return ret_df
    
    def profile(self, engine_type = 'pandas'):
        if isinstance(self.auto_pipeline['profiler'], type(None)):
            return "feature profile support for multiple table is WIP"
        return self.auto_pipeline['profiler'].visualize_analyze(engine_type)

    def feature_importance(self):
        import pandas as pd
        if isinstance(self.auto_pipeline['estimator'], type(None)):
            return "feature estimator support for multiple table is WIP"
        fe_imp_dict = self.auto_pipeline['estimator'].get_feature_importance()
        feat_importances = pd.Series([i[1] for i in fe_imp_dict], [i[0] for i in fe_imp_dict])
        feat_importances = feat_importances[feat_importances > 0].sort_values()
        height = int(len(feat_importances) * 0.5)
        return feat_importances.plot(kind='barh', figsize=(15,height))

    def plot(self):
        if self.auto_pipeline['relational']:
            return self.auto_pipeline['relational'].plot()
        else:
            return self.auto_pipeline['estimator'].plot()
        
    def get_transformed_data(self):
        if self.auto_pipeline['relational']:
            return self.auto_pipeline['relational'].get_transformed_cache()
        else:
            ret_df = self.auto_pipeline['estimator'].get_transformed_cache()
            return ret_df
        
    def get_feature_list(self):
        fe_imp_dict = self.auto_pipeline['estimator'].get_feature_importance()
        feature_list = [i[0] for i in fe_imp_dict if i[1] > 0]
        return feature_list
    
    def add_operation(self, config):
        if self.auto_pipeline['relational']:
            return self.auto_pipeline['relational'].add_operation(config)
        else:
            return self.auto_pipeline['estimator'].add_operation(config)
        
    def delete_operation(self, idx):
        if self.auto_pipeline['relational']:
            return self.auto_pipeline['relational'].delete_operation(idx)
        else:
            return self.auto_pipeline['estimator'].delete_operation(idx)