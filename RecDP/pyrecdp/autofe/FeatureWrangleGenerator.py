import logging
from pyrecdp.widgets.utils import Timer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.utils import infer_problem_type

from pyrecdp.widgets import BaseWidget, TabWidget

import pandas as pd
import os
import yaml
from pandas_profiling import ProfileReport
from pandas_profiling import config
dir_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')

logger = logging.getLogger(__name__)


class TabularPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self, df_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_generator_with_recdp()
 
    def _replace_generator_with_recdp(self):        
        new_generators = []
        for i in range(len(self.generators)):
            new_sub_generators = []
            for generator in self.generators[i]:
                new_cls = self._get_pyrecdp_class(generator)
                if new_cls:
                    new_sub_generators.append(new_cls)
            new_generators.append(new_sub_generators)
        self.generators = new_generators
        
    def _get_pyrecdp_class(self, obj):
        from pyrecdp.primitives.generators import cls_list
        cls_name = obj.__class__.__name__
        if cls_name in cls_list:
            return cls_list[cls_name](obj)
        else:
            return None
        
class TabularAnalyzer:
    pass


class FeatureWrangleGenerator:
    def __init__(self, dataset, label, only_pipeline = False, *args, **kwargs):
        self.label = label
        self.data = dataset

        # prepare main view
        self.original_df_view = BaseWidget(display_flag=False)
        self.original_df_profile_view = BaseWidget(display_flag=False)
        self.log_view = BaseWidget(display_flag=False)
        tab_children = [('log', self.log_view), ('original_df', self.original_df_view), ('profiler-orginal_df', self.original_df_profile_view)]
        self.main_view = TabWidget(tab_children)
        Timer.viewer = self.log_view
        pdp_config = config.Settings().parse_obj(yaml.safe_load(open(f"{dir_path}/../widgets/pandas_profiling_config.yaml")))
        pdp_config.interactions.targets = [label]

        with Timer("FeatureWrangleGenerator Data Wrangling"):
            # detect problem type
            with Timer("Detecting problem type"):
                self.problem_type = infer_problem_type(y=self.data[self.label], silent=False)
                self.log_view.display(f"Detected Problem Type is: {self.problem_type}")

            # profile original dataframe
            self.original_df_view.display(self.data)
            with Timer("Profiling original dataframe"):
                self.original_df_profile = ProfileReport(self.data, title="Original DataFrame Profiling", config=pdp_config)
            self.original_df_profile_view.display(self.original_df_profile) 

            # create feature engineering pipline
            self.feature_generator = TabularPipelineFeatureGenerator(len(self.data))
            if not only_pipeline:
                # auto feature engineering
                with Timer("Auto Feature Engineering on dataset"):
                    self.transformed_feature = self.feature_generator.fit_transform(self.data, y=self.data[self.label])
                # create view for transformed data
                self.transformed_df_view = BaseWidget(display_flag=False)
                self.transformed_df_profile_view = BaseWidget(display_flag=False)
                self.main_view.append('transformed_df', self.transformed_df_view)
                self.main_view.append('profiler-transformed_df', self.transformed_df_profile_view)

                # profile transformed dataframe
                self.transformed_df_view.display(self.get_transformed_data())
                with Timer("profiling transformed dataset"):
                    self.transformed_df_profile = ProfileReport(self.get_transformed_data(), title="Transformed DataFrame Profiling", config=pdp_config)
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
