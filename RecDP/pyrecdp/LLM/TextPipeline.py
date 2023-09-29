from pyrecdp.core import DiGraph
from pyrecdp.core.pipeline import BasePipeline
from pyrecdp.primitives.operations import Operation, BaseOperation
from pyrecdp.primitives.operations.ray_dataset import RayDatasetReader
import logging
from pyrecdp.core.utils import Timer, deepcopy
from IPython.display import display
from tqdm import tqdm
import types
from ray.data import Dataset
import ray

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class TextPipeline(BasePipeline):
    def __init__(self, pipeline_file=None):
        super().__init__()
        if pipeline_file != None:
            self.import_from_yaml(pipeline_file)
        else:
            #add a data set input place holder
            op = RayDatasetReader()
            self.add_operation(op)
            
    def __del__(self):
        if ray.is_initialized():
            ray.shutdown()
            
    def execute(self, ds: Dataset = None) -> Dataset:
        # prepare pipeline
        if not hasattr(self, 'executable_pipeline') or not hasattr(self, 'executable_sequence'):
            self.executable_pipeline, self.executable_sequence = self.create_executable_pipeline()
        executable_pipeline = self.executable_pipeline
        executable_sequence = self.executable_sequence
        
        print("init ray")
        if not ray.is_initialized():
            ray.init()

        # execute
        with Timer(f"execute with ray"):
            for op in executable_sequence:
                if ds != None and isinstance(op, RayDatasetReader):
                    op.cache = ds
                else:
                    op.execute_ray(executable_pipeline)
            if len(executable_sequence) > 0:
                ds = executable_sequence[-1].cache
                if isinstance(ds, Dataset):
                    ds = ds.materialize()
        
        # fetch result
        return ds
    
    def add_operation(self, config):        
        # get current max operator id
        max_idx = self.pipeline.get_max_idx()
        cur_idx = max_idx + 1
        find_children_skip = False
        
        if not isinstance(config, dict):
            op = config
            if max_idx == -1:
                leaf_child = None
            else:
                pipeline_chain = self.to_chain()
                leaf_child = [pipeline_chain[-1]]
            
            config = {
                "children": leaf_child,
                "inline_function": op,
            }
            find_children_skip = True
        children = config["children"]
        inline_function = config["inline_function"]
        
        if not isinstance(children, list) and children is not None:
            children = [children]
        
        # ====== Start to add it to pipeline ====== #
        if isinstance(inline_function, types.FunctionType):
            config = {
                "func_name": inline_function,
            }
            self.pipeline[cur_idx] = Operation(
                cur_idx, children, output = None, op = "ray_python", config = config)
        elif isinstance(inline_function, BaseOperation):
            op_name = inline_function.op.op
            #config = vars(inline_function)
            config = inline_function.op.config
            self.pipeline[cur_idx] = Operation(
                cur_idx, children, output = None, op = op_name, config = config)
        
        # we need to find nexts
        if find_children_skip:
            return self.pipeline
        for to_replace_child in children:
            next = []
            for idx in pipeline_chain:
                if self.pipeline[idx].children and to_replace_child in self.pipeline[idx].children:
                    next.append(idx)
            for idx in next:
                # replace next's children with new added operator
                children_in_next = self.pipeline[idx].children
                found = {}
                for id, child in enumerate(children_in_next):
                    if child == to_replace_child:
                        found[id] = cur_idx
                for k, v in found.items():
                    self.pipeline[idx].children[k] = v
        return self.pipeline
                    
    def add_operations(self, config_list):
        for op in config_list:
            self.add_operation(op)
        return self.pipeline
     
    def profile(self):
        # TODO: print analysis and log for each component.
        pass
            