from pyrecdp.primitives.generators import *
from pyrecdp.core import DataFrameAPI, DataFrameSchema, SparkDataProcessor
import pandas as pd
import numpy as np
import logging
import time

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class FeatureWrangler():
    def __init__(self, dataset, label, *args, **kwargs):
        X = DataFrameAPI.instiate(dataset)
        if isinstance(label, str):
            if label not in dataset.columns:
                raise ValueError(f"label {label} is not found in dataset")
            y = dataset[label]
        else:
            y = label
        to_select = [i for i in X.columns if i != y.name]
        self.feature_data = X[to_select]
        self.y = y
        
        # add default pipeline
        self.generators = []
        self.generators.append([DataframeConvertFeatureGenerator()])
        self.generators.append([cls() for cls in pre_feature_generator_list])
        self.generators.append([cls() for cls in transformation_generator_list])
        self.generators.append([DataframeTransformFeatureGenerator()])
        self.generators.append([cls() for cls in encode_generator_list])
        self.generators.append([cls() for cls in index_generator_list])
        self.generators.append([cls() for cls in post_feature_generator_list])

        self.fit_analyze()

    
    def fit_analyze(self, *args, **kwargs):
        # Chendi: Since fit_analyze is mainly focusing on decide which primitives we should use, avoid feeding too big dataframe here
        # If data size is over 10,000, do sample here
        sampled_feature_data = self.feature_data.may_sample()
        cur_feature_list = DataFrameSchema(sampled_feature_data)
        for i in range(len(self.generators)):
            generator_group_valid = []
            for generator in self.generators[i]:
                if generator.is_useful(cur_feature_list):
                    if isinstance(generator, TypeInferFeatureGenerator):
                        cur_feature_list = generator.fit_prepare(cur_feature_list, sampled_feature_data)
                    else:
                        cur_feature_list = generator.fit_prepare(cur_feature_list)
                    generator_group_valid.append(generator)
            self.generators[i] = generator_group_valid

    def display_transform_pipeline(self):
        return [f"Stage {i}: {[g.__class__ for g in stage]}" for i, stage in enumerate(self.generators)]

    def generate_pipeline_code(self, engine_type = "pandas", *args, **kwargs):
        if engine_type == "spark":
            return self._generate_pipeline_code_spark(*args, **kwargs)
        else:
            return self._generate_pipeline_code_pd(*args, **kwargs)

    def fit_transform(self, engine_type = 'pandas', *args, **kwargs):
        func_chain = self.generate_pipeline_code(engine_type)
        ret = self.feature_data
        for func in func_chain:
            start_time = time.time()
            ret = func(ret)
            end_time = time.time()
            if engine_type == "pandas":
                print(f"Transformation of {func} took {(end_time - start_time):.3f} secs")
        return ret

    def dump_pipeline_codes(self):
        for generator_stage in self.generators:
            for generator in generator_stage:
                print(generator.dump_codes())
        
    def _generate_pipeline_code_pd(self, *args, **kwargs):
        func_chain = []
        for generator_stage in self.generators:
            for generator in generator_stage:
                func_chain.append(generator.get_function_pd())
        return func_chain
    
    def _generate_pipeline_code_spark(self, *args, **kwargs):
        rdp = SparkDataProcessor()
        func_chain = []
        for generator_stage in self.generators:
            for generator in generator_stage:
                func_chain.append(generator.get_function_spark(rdp))
        return func_chain

