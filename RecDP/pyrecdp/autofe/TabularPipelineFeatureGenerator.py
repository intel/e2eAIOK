from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from pyrecdp.primitives.generators import *
import pandas as pd
import numpy as np

class TabularPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self, dataset, label, *args, **kwargs):
        X = dataset
        if isinstance(label, str):
            if label not in dataset.columns:
                raise ValueError(f"label {label} is not found in dataset")
            y = dataset[label]
        else:
            y = label
        to_select = [i for i in X.columns if i != y.name]
        self.feature_data = X[to_select]
        self.y = y

        super().__init__(*args, **kwargs)
        # TODO remove below
        self._replace_generator_with_recdp()
        
    def fit_analyze(self, *args, **kwargs):
        self._infer_features_in_full(self.feature_data)
        feature_metadata = self.feature_metadata_in
        
        for i in range(len(self.generators)):
            generator_group_valid = []
            for generator in self.generators[i]:
                if generator.is_valid_metadata_in(feature_metadata):
                    generator_group_valid.append(generator)
            self.generators[i] = generator_group_valid

    def fit_transform(self, *args, **kwargs):
        return super().fit_transform(self.feature_data, self.y, *args, **kwargs)

    def display_transform_pipeline(self):
        return [f"Stage {i}: {[g.__class__ for g in stage]}" for i, stage in enumerate(self.generators)]

    def _get_default_generators(self, vectorizer=None):
        generators = super()._get_default_generators(vectorizer)
        return generators
    
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
            # check if already pyrecdp class
            if isinstance(obj, cls_list[cls_name]):
                return obj
            return cls_list[cls_name](obj)
        else:
            return None