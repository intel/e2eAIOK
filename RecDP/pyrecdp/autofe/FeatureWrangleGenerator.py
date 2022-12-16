import logging
from pyrecdp.widgets.utils import Timer
from autogluon.core.utils import infer_problem_type

from pyrecdp.widgets import BaseWidget, TabWidget
from pyrecdp.autofe import FeatureProfiler, TabularPipelineFeatureGenerator

import os

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class FeatureWrangleGenerator:
    def __init__(self, dataset, label, only_pipeline = False, enable_profiler = True, *args, **kwargs):
        self.label = label
        self.data = dataset

        # prepare main view
        self.original_df_view = BaseWidget(display_flag=False)
        self.log_view = BaseWidget(display_flag=False)
        tab_children = [('log', self.log_view), ('original_df', self.original_df_view)]
        self.main_view = TabWidget(tab_children)
        Timer.viewer = self.log_view

        # prepare pipeline        
        self.feature_generator = TabularPipelineFeatureGenerator(self.data, self.data[self.label])

        # start data analysis and transform
        with Timer("FeatureWrangleGenerator Data Wrangling"):
            # detect problem type
            with Timer("Detecting problem type"):
                self.problem_type = infer_problem_type(y=self.data[self.label], silent=False)
                self.log_view.display(f"Detected Problem Type is: {self.problem_type}")

            # profile original dataframe
            self.original_df_view.display(self.data)
            with Timer("Analysis Original DataFrame"):
                self.feature_generator.fit_analyze()
                self.log_view.display(f"feature engineering pipeline after analysis:\n{self.get_transform_pipeline()}")
            with Timer("Prepare profiling view"):
                self.original_df_profile = FeatureProfiler(self.data, self.label).visualize_analyze()
                self.original_df_profile_view = BaseWidget(display_flag=False)
                self.main_view.append('profiler-original', self.original_df_profile_view)
                self.original_df_profile_view.display(self.original_df_profile)

            # auto feature engineering
            with Timer("Auto Feature Engineering on dataset"):
                self.transformed_feature = self.feature_generator.fit_transform()

            # create view for transformed data
            self.transformed_df_view = BaseWidget(display_flag=False)
            self.main_view.append('transformed_df', self.transformed_df_view)
            self.transformed_df_view.display(self.get_transformed_data())

            # profile transformed dataframe
            with Timer("profiling transformed dataset"):
                self.transformed_df_profile = FeatureProfiler(self.transformed_feature, self.data[self.label]).visualize_analyze()
                self.transformed_df_profile_view = BaseWidget(display_flag=False)
                self.main_view.append('profiler-transformed', self.transformed_df_profile_view)
                self.transformed_df_profile_view.display(self.transformed_df_profile)
            

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
