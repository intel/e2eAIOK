from pyrecdp.primitives.generators import *
from pyrecdp.primitives.profilers import *
from pyrecdp.autofe.TabularPipeline import TabularPipeline
import logging
from pyrecdp.core.dataframe import DataFrameAPI
from pyrecdp.core.schema import SeriesSchema
import pandas as pd
import copy

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class FeatureWrangler(TabularPipeline):
    def __init__(self, dataset=None, label=None, data_pipeline=None, time_series = None, exclude_op = [], include_op = [], *args, **kwargs):
        if data_pipeline is None:
            super().__init__(dataset, label, exclude_op, include_op)
            self.data_profiler = [cls() for cls in feature_infer_list]
        else:
            self.generators = []
            self.data_profiler = []
            self.pipeline = data_pipeline.pipeline.copy()
            self.dataset = data_pipeline.dataset
            self.ts = time_series
            self.include_op = include_op
            self.exclude_op = exclude_op
            self.rdp = data_pipeline.rdp
            self.transformed_cache = data_pipeline.transformed_cache if hasattr(data_pipeline, 'transformed_cache') else None
            self.y = data_pipeline.y
            self.main_table = data_pipeline.main_table
        # If we provided multiple datasets in this workload
        self.generators.append([cls() for cls in pre_feature_generator_list])
        self.generators.append([cls() for cls in transformation_generator_list])
        self.generators.append([cls() for cls in pre_enocode_feature_generator_list])
        self.generators.append([cls() for cls in local_encode_generator_list])
        self.generators.append([cls() for cls in global_dict_index_generator_list])
        self.generators.append([cls() for cls in post_feature_generator_list])
        self.generators.append([cls(final = True) for cls in final_generator_list])

        # add a default exclude list for some not very useful but performance impact FE
        default_exclude_op = ['TargetEncodeFeatureGenerator']
        # default_exclude_op = []
        for op_name in default_exclude_op:
            if op_name not in self.include_op:
                self.exclude_op.append(op_name)
        print(f"We exclude some Feature Engineer Generator as listed, you can use 'include_op = [\"XXXFeatureGenerator\"]'  to re-add them, exclude_op list {self.exclude_op}")

        self.ts = time_series
        self.fit_analyze()

    def fit_analyze(self, *args, **kwargs): 
        child = list(self.pipeline.keys())[-1]
        max_id = child
        # sample data
        X = DataFrameAPI().instiate(self.dataset[self.main_table])
        sampled_data = X.may_sample()
        
        for generator in self.data_profiler:
            self.pipeline, child, max_id = generator.fit_prepare(self.pipeline, [child], max_id, sampled_data, self.y, self.ts)
        print("Feature List generated, using analyzed feature tags to create data pipeline")
        ret = super().fit_analyze(*args, **kwargs)
        self.update_label()
        return ret