import logging
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.utils import infer_problem_type
import pandas as pd

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

        # detect problem type
        self.problem_type = infer_problem_type(y=self.data[self.label], silent=False)

        # generate feature engineering pipline
        self.feature_generator = TabularPipelineFeatureGenerator(len(self.data))
        if not only_pipeline:
            self.transformed_feature = self.feature_generator.fit_transform(self.data, y=self.data[self.label])

    def get_transform_pipeline(self):
        return "\n".join([f"Stage {i}: {[g.__class__ for g in stage]}" for i, stage in enumerate(self.feature_generator.generators)])

    def get_feature_list(self):
        return self.transformed_feature.dtypes
    
    def get_transformed_data(self):
        return self.transformed_feature
    
    def get_original_data(self):
        return self.data

    def get_problem_type(self):
        return self.problem_type
