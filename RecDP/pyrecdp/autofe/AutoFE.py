"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import logging
from pyrecdp.core.utils import Timer, infer_problem_type
from pyrecdp.core.dataframe import DataFrameAPI

from pyrecdp.autofe import FeatureWrangler, FeatureProfiler, RelationalBuilder, FeatureEstimator, TabularPipeline

import os

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class AutoFE():
    def __init__(self, dataset, label, time_series = None, exclude_op = [], include_op = [], *args, **kwargs):
        self.label = label
        self.auto_pipeline = {'relational': None, 'profiler': None, 'wrangler': None, 'estimator': None}
        if isinstance(dataset, dict):
            print("AutoFE started to build data relation")
            self.auto_pipeline['relational'] = RelationalBuilder(dataset=dataset, label=label)
            pipeline = self.auto_pipeline['relational']
            print("AutoFE started to create data pipeline")
            self.auto_pipeline['wrangler'] = FeatureWrangler(data_pipeline=pipeline, exclude_op = exclude_op, include_op = include_op, time_series = time_series)
            pipeline = self.auto_pipeline['wrangler']
            config = {
                'model_file': 'autofe_lightgbm.mdl',
                'objective': infer_problem_type(pipeline.dataset[pipeline.main_table], label),
                'model_name': 'lightgbm'}
            self.auto_pipeline['estimator'] = FeatureEstimator(data_pipeline = pipeline, config = config)
        else:
            print("AutoFE started to profile data")
            self.auto_pipeline['profiler'] = FeatureProfiler(dataset=dataset, label=label)
            print("AutoFE started to create data pipeline")
            self.auto_pipeline['wrangler'] = FeatureWrangler(dataset=dataset, label=label, time_series = time_series, exclude_op = exclude_op, include_op = include_op)
            pipeline = self.auto_pipeline['wrangler']
            config = {
                'model_file': 'autofe_lightgbm.mdl',
                'objective': infer_problem_type(pipeline.dataset[pipeline.main_table], label),
                'model_name': 'lightgbm'}
            self.auto_pipeline['estimator'] = FeatureEstimator(data_pipeline = pipeline, config = config)

    def fit_transform(self, engine_type = 'pandas', no_cache = False, *args, **kwargs):
        print("AutoFE started to fit_transform data")
        ret_df = None
        pipeline = self.auto_pipeline['estimator']
        ret_df = pipeline.fit_transform(engine_type, data = ret_df, **kwargs)
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
        if not isinstance(fe_imp_dict, dict):
            return "This dataset doesn't have feature importance estimator support"
        feat_importances = pd.Series([i[1] for i in fe_imp_dict], [i[0] for i in fe_imp_dict])
        feat_importances = feat_importances[feat_importances > 0].sort_values()
        height = int(len(feat_importances) * 0.5)
        try:
            ret = feat_importances.plot(kind='barh', figsize=(15,height))
        except:
            ret = feat_importances
        return ret

    def plot(self):
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

    def export(self, file_path = None):
        return self.auto_pipeline['wrangler'].export(file_path)

    @classmethod
    def clone_pipeline(cls, origin_pipeline, data):
        pipeline_json = origin_pipeline.export()
        new_pipeline = TabularPipeline(data, origin_pipeline.label)
        new_pipeline.import_from_json(pipeline_json)
        return new_pipeline