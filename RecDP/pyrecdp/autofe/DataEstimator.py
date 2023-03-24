from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
import logging
from pyrecdp.primitives.operations import Operation
from pyrecdp.core import DiGraph
import copy

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class DataEstimator(BasePipeline):
    def __init__(self, objective, metrics, model_name, method = 'predict', model_file = None, dataset = None, label = None, data_pipeline = None):
        if dataset and label:
            super().__init__(dataset, label)
            if isinstance(data_pipeline, DiGraph):
                self.pipeline = data_pipeline
            elif isinstance(data_pipeline, str):
                self.import_from_json(data_pipeline)
        else:
            if isinstance(data_pipeline, BasePipeline):
                self.pipeline = data_pipeline.pipeline
                self.dataset = data_pipeline.dataset
                self.rdp = data_pipeline.rdp
                label = data_pipeline.y.name if label is None else label
            else:
                raise NotImplementedError(f"Unsupport input datapipeline is {data_pipeline}")
        if not label:
            raise ValueError("Unable to find label for this pipeline, please provide it through API")
        max_idx = self.pipeline.get_max_idx()
        leaf_idx = self.pipeline.convert_to_node_chain()[-1]
        
        cur_idx = max_idx + 1
        if model_name == 'lightgbm':
            if not model_file:
                model_file = f"lightgbm_{objective}_{metrics}_{label}.mdl"              
                
        config = {'label': label, 'metrics': metrics, 'objective': objective, 'model_file': model_file, 'method': method}
        op = Operation(cur_idx, [leaf_idx], None, model_name, config)
        self.pipeline[cur_idx] = op

        

    