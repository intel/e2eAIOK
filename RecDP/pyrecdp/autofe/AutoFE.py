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
        self.auto_pipeline = {'relational': None, 'wrangler': None, 'estimator': None}
        if isinstance(dataset, dict):
            self.auto_pipeline['relational'] = RelationalBuilder(dataset=dataset, label=label)
        else:
            print("AutoFE started to profile data")
            self.auto_pipeline['profiler'] = FeatureProfiler(dataset=dataset, label=label)
            print("AutoFE started to create data pipeline")
            self.auto_pipeline['wrangler'] = FeatureWrangler(dataset=dataset, label=label)

        engine_type = 'pandas'

        if self.auto_pipeline['relational']:
            ret_df = {}
            for k, v in ret_df.items():
                X = DataFrameAPI().instiate(self.dataset[k])
                ret_df[k] = X.may_sample()
            pipeline = self.auto_pipeline['relational']
            ret_df = pipeline.fit_transform(engine_type)
            self.auto_pipeline['relational'] = RelationalBuilder(dataset=ret_df, label=self.label)

        if self.auto_pipeline['wrangler']:
            pipeline = self.auto_pipeline['wrangler']
            config = {
            'model_file': 'autofe_lightgbm.mdl',
            'objective': infer_problem_type(pipeline.dataset[pipeline.main_table][label]),
            'model_name': 'lightgbm'}
            self.auto_pipeline['estimator'] = FeatureEstimator(data_pipeline = pipeline, config = config)

    def fit_transform(self, engine_type = 'pandas', no_cache = False, *args, **kwargs):
        ret_df = None
        if self.auto_pipeline['relational']:
            pipeline = self.auto_pipeline['relational']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)

        # if self.auto_pipeline['wrangler']:
        #     pipeline = self.auto_pipeline['wrangler']
        #     ret_df = pipeline.fit_transform(engine_type, data = ret_df)

        if self.auto_pipeline['estimator']:
            pipeline = self.auto_pipeline['estimator']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)
        return ret_df
    
    def profile(self, engine_type):
        return self.auto_pipeline['profiler'].visualize_analyze(engine_type)

    def feature_importance(self):
        import pandas as pd
        fe_imp_dict = self.auto_pipeline['estimator'].get_feature_importance()
        feat_importances = pd.Series([i[1] for i in fe_imp_dict], [i[0] for i in fe_imp_dict])
        return feat_importances.plot(kind='barh')

    def plot(self):
        return self.auto_pipeline['estimator'].plot()        
        
    def get_transformed_cache(self):
        return self.auto_pipeline['estimator'].get_transformed_cache()