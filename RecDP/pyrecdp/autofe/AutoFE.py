import logging
from pyrecdp.core.utils import Timer
from pyrecdp.core.dataframe import DataFrameAPI

from pyrecdp.autofe import FeatureWrangler, RelationalBuilder, FeatureEstimator

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
            'metrics': 'auc',
            'objective': 'binary',
            'model_name': 'lightgbm'}
            self.auto_pipeline['estimator'] = FeatureEstimator(method = 'train', data_pipeline = pipeline, config = config)

    def transform(self, engine_type = 'pandas', no_cache = False, *args, **kwargs):
        if self.auto_pipeline['relational']:
            pipeline = self.auto_pipeline['relational']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)

        if self.auto_pipeline['wrangler']:
            pipeline = self.auto_pipeline['wrangler']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)

        if self.auto_pipeline['estimator']:
            pipeline = self.auto_pipeline['estimator']
            ret_df = pipeline.fit_transform(engine_type, data = ret_df)

        return ret_df
