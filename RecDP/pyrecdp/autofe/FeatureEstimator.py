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

from pyrecdp.primitives.generators import *
from pyrecdp.autofe.TabularPipeline import TabularPipeline
import logging
from pyrecdp.primitives.operations import Operation
from pyrecdp.core.di_graph import DiGraph
import copy
from pyrecdp.data_processor import DataProcessor as SparkDataProcessor

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class FeatureEstimator(TabularPipeline):
    def __init__(self, data_pipeline, config = None):
        """
        Parameters: data_pipeline, method and config
        config = {
            'model_file': string,
            'dataset': path or dataframe,
            'label': string,
            'model_name': string,
            'objective': string,
            'train_test_splitter': function
        }
        Two option:
        data_pipeline is json_file_path, specify config for ['dataset', 'label']
        data_pipeline is pre created pipeline object, specify config for ['model_name', 'objective']
        """
        method = 'train'
        model_file = config['model_file'] if 'model_file' in config else None
        label = config['label'] if 'label' in config else None
        self.transformed_cache = None
        if isinstance(data_pipeline, str):
            dataset = config['dataset'] 
            if dataset is not None:
                super().__init__(dataset, label)
                self.import_from_json(data_pipeline)
        elif isinstance(data_pipeline, TabularPipeline):
            self.nested_pipeline_obj = data_pipeline
            self.pipeline = data_pipeline.pipeline.copy()
            self.dataset = data_pipeline.dataset
            self.rdp = data_pipeline.rdp
            self.transformed_cache = data_pipeline.transformed_cache if hasattr(data_pipeline, 'transformed_cache') else None
            label = data_pipeline.y if label is None else label
            self.y = label
        else:
            raise NotImplementedError(f"Unsupport input datapipeline is {data_pipeline}")


        max_idx = self.pipeline.get_max_idx()
        leaf_idx = self.pipeline.convert_to_node_chain()[-1]
        self.transformed_end_idx = max_idx
        
        if label is None and method != 'predict':
            return

        if self.pipeline[leaf_idx].op not in ["lightgbm"]:
            model_name = config['model_name']
            objective = config['objective']
            if objective == 'binary':
                config['metrics'] = 'auc'
            elif objective == 'regression':
                config['metrics'] = 'rmse'
            if method == 'train':
                train_test_splitter = config['train_test_splitter'] if 'train_test_splitter' in config else None
            if model_file is None:
                model_file = f"{model_name}_{objective}_{label}.mdl"
            cur_idx = max_idx + 1
            self.estimator_pipeline_start = cur_idx
            # we need to add two op, one to prepare dataset, one for estimator
            op = Operation(cur_idx, [leaf_idx], None, 'DataFrame', 'main_table')
            self.pipeline[cur_idx] = op
            self.transformed_end_idx = cur_idx
            
            child_idx = cur_idx
            cur_idx += 1
            op_config = {'label': label, 'objective': objective, 'train_test_splitter': train_test_splitter}
            if 'metrics' in config:
                op_config['metrics'] = config['metrics']
            op = Operation(cur_idx, [child_idx], None, model_name, op_config)
            self.pipeline[cur_idx] = op
        else:
            self.pipeline[leaf_idx].config['method'] = method  

    def fit_transform(self, engine_type = 'pandas', no_cache = False, data = None, *args, **kwargs):
        if self.transformed_cache is not None and not no_cache: # we can skip data process steps
            start_op_idx = self.estimator_pipeline_start
        else:
            start_op_idx = -1
        if engine_type == "spark":
            if "spark_master" in kwargs:
                spark_master = kwargs["spark_master"]
                spark_mode = 'standalone'
            else:
                spark_master = "local[*]"
                spark_mode = 'local'
            self.rdp = SparkDataProcessor(spark_mode=spark_mode, spark_master=spark_master)
        ret = self.execute(engine_type, start_op_idx, no_cache, data = data, transformed_end = self.transformed_end_idx)
        if engine_type == "spark":
            del self.rdp 
            self.rdp = None
        if self.transformed_cache is None:
            self.transformed_cache = ret
        self.feature_importance = self.executable_sequence[-1].cache
        return ret
    
    def get_feature_importance(self):
        return self.feature_importance