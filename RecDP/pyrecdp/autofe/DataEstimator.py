from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
import logging
from pyrecdp.primitives.operations import Operation
from pyrecdp.core import DiGraph
import copy

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class DataEstimator(BasePipeline):
    def __init__(self, data_pipeline, method, config = None):
        """
        Parameters: data_pipeline, method and config
        config = {
            'model_file': string,
            'dataset': path or dataframe,
            'label': string,
            'model_name': string,
            'objective': string,
            'metrics': string,
            'train_test_splitter': function
        }
        Two option:
        data_pipeline is json_file_path, specify config for ['dataset', 'label']
        data_pipeline is pre created pipeline object, specify config for ['model_name', 'objective', 'metrics']
        """
        model_file = config['model_file'] if 'model_file' in config else None
        label = config['label'] if 'label' in config else None
        if isinstance(data_pipeline, str):
            dataset = config['dataset'] 
            if dataset is not None:
                super().__init__(dataset, label)
                self.import_from_json(data_pipeline)
        elif isinstance(data_pipeline, BasePipeline):
            self.pipeline = data_pipeline.pipeline
            self.dataset = data_pipeline.dataset
            self.rdp = data_pipeline.rdp
            label = data_pipeline.y if label is None else label
        else:
            raise NotImplementedError(f"Unsupport input datapipeline is {data_pipeline}")
        if label is None and method != 'predict':
            raise ValueError("Unable to find label for this pipeline, please provide it through API")
        
        max_idx = self.pipeline.get_max_idx()
        leaf_idx = self.pipeline.convert_to_node_chain()[-1]
        if self.pipeline[leaf_idx].op not in ["lightgbm"]:
            model_name = config['model_name']
            objective = config['objective']
            metrics = config['metrics']
            if method == 'train':
                train_test_splitter = config['train_test_splitter'] if 'train_test_splitter' in config else None
            if model_file is None:
                model_file = f"{model_name}_{objective}_{metrics}_{label}.mdl"
            cur_idx = max_idx + 1
            op_config = {'label': label, 'metrics': metrics, 'objective': objective, 'model_file': model_file, 'method': method, 'train_test_splitter': train_test_splitter}
            op = Operation(cur_idx, [leaf_idx], None, model_name, op_config)
            self.pipeline[cur_idx] = op
        else:
            self.pipeline[leaf_idx].config['method'] = method

        

    