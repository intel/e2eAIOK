from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
import logging
from pyrecdp.primitives.operations import Operation
from pyrecdp.core import DiGraph
import copy
from pyrecdp.core import SparkDataProcessor

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class FeatureEstimator(BasePipeline):
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
        elif isinstance(data_pipeline, BasePipeline):
            self.nested_pipeline_obj = data_pipeline
            self.pipeline = data_pipeline.pipeline
            self.dataset = data_pipeline.dataset
            self.rdp = data_pipeline.rdp
            self.transformed_cache = data_pipeline.transformed_cache if hasattr(data_pipeline, 'transformed_cache') else None
            label = data_pipeline.y if label is None else label
        else:
            raise NotImplementedError(f"Unsupport input datapipeline is {data_pipeline}")
        if label is None and method != 'predict':
            raise ValueError("Unable to find label for this pipeline, please provide it through API")
        
        max_idx = self.pipeline.get_max_idx()
        leaf_idx = self.pipeline.convert_to_node_chain()[-1]
        self.transformed_end_idx = -1
        
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
            self.rdp = SparkDataProcessor()
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