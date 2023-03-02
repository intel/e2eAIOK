import logging
from pyrecdp.core.utils import Timer

from pyrecdp.widgets import BaseWidget, TabWidget
from pyrecdp.autofe import FeatureProfiler, FeatureWrangler

import os

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class AutoFE:
    def __init__(self, dataset, label, only_pipeline = False, enable_profiler = True, *args, **kwargs):
        self.label = label
        self.data = dataset

        # prepare main view
        self.original_df_view = BaseWidget(display_flag=False)
        self.log_view = BaseWidget(display_flag=False)
        self.main_view = TabWidget([('log', self.log_view), ('original_df', self.original_df_view)])
        Timer.viewer = self.log_view

        # prepare pipeline        
        self.feature_generator = FeatureWrangler(self.data, self.data[self.label])

        # start data analysis and transform
        with Timer("AutoFE Data Wrangling"):
            # profile original dataframe
            self.original_df_view.display(self.data)
            with Timer("Analysis Original DataFrame"):
                self.feature_generator.fit_analyze()
                self.log_view.display(f"feature engineering pipeline after analysis:\n{self.get_transform_pipeline()}")

            # auto feature engineering
            with Timer("Auto Feature Engineering on dataset"):
                self.transformed_feature = self.feature_generator.fit_transform()

            # create view for transformed data
            self.transformed_df_view = BaseWidget(display_flag=False)
            self.main_view.append('transformed_df', self.transformed_df_view)
            self.transformed_df_view.display(self.get_transformed_data())

            # profile transformed dataframe
            with Timer("profiling transformed dataset"):
                self.transformed_df_profile = FeatureProfiler(self.transformed_feature, self.data[self.label]).visualize_analyze().show()
            

    def get_transform_pipeline(self):
        return "\n".join([f"Stage {i}: {[g.__class__ for g in stage]}" for i, stage in enumerate(self.feature_generator.generators)])
    
    def exclude_target(self, df):
        label = self.label if isinstance(self.label, list) else [self.label]
        feat_columns = [n for n in df.columns if n not in label]
        return df[feat_columns]
    
    def get_origin_feature_list(self):
        return self.exclude_target(self.data).dtypes

    def get_feature_list(self):
        return self.exclude_target(self.transformed_feature).dtypes
    
    def get_transformed_data(self):
        return self.transformed_feature
    
    def get_original_data(self):
        return self.data

    def get_problem_type(self):
        return self.problem_type
