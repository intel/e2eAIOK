from pyrecdp.core import DiGraph
from pyrecdp.core.pipeline import BasePipeline
from pyrecdp.primitives.operations import Operation, BaseOperation
from pyrecdp.primitives.operations.text_reader import DatasetReader, PerfileReader
from pyrecdp.primitives.operations.text_writer import PerfileParquetWriter
import logging
from pyrecdp.core.utils import Timer, deepcopy
from IPython.display import display
from tqdm import tqdm
import types
from ray.data import Dataset
from pyspark.sql import DataFrame
import ray
from pyrecdp.core import SparkDataProcessor
import time
import os
import psutil

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

total_mem = int(psutil.virtual_memory().total * 0.6)
total_cores = psutil.cpu_count(logical=False)

class TextPipeline(BasePipeline):
    def __init__(self, pipeline_file=None):
        super().__init__()
        if pipeline_file != None:
            self.import_from_yaml(pipeline_file)
        else:
            #add a data set input place holder
            op = DatasetReader()
            self.add_operation(op)
            
    def __del__(self):
        if hasattr(self, 'engine_name') and self.engine_name == 'ray':
            if ray.is_initialized():
                ray.shutdown()
            
    def check_platform(self, executable_sequence):
        is_spark = True
        is_ray = True
        spark_list = []
        ray_list = []
        for op in executable_sequence:
            is_spark = op.support_spark if is_spark else False
            is_ray = op.support_ray if is_ray else False
            if op.support_ray:
                ray_list.append(str(op))
            if op.support_spark:
                spark_list.append(str(op))
        if is_ray:
            return 'ray'
        elif is_spark:
            return 'spark'
        else:
            print(f"We can't identify an uniform engine for this pipeline. \n  Operations work on Ray are {ray_list}. \n  Operations work on Spark are {spark_list}")
            return 'mixed'
            
    def execute(self, ds = None):
        # prepare pipeline
        if not hasattr(self, 'executable_pipeline') or not hasattr(self, 'executable_sequence'):
            self.executable_pipeline, self.executable_sequence = self.create_executable_pipeline()
        executable_pipeline = self.executable_pipeline
        executable_sequence = self.executable_sequence
        
        engine_name = self.check_platform(executable_sequence)
        
        if engine_name == 'ray':
            print("init ray")
            if not ray.is_initialized():
                print(f"init ray with total mem of {total_mem}, total core of {total_cores}")
                try:
                    ray.init(object_store_memory=total_mem, num_cpus=total_cores)
                except:
                    ray.init()

            # execute
            with Timer(f"execute with ray"):
                for op in executable_sequence:
                    if ds != None and isinstance(op, DatasetReader):
                        op.cache = ds
                    else:
                        op.execute_ray(executable_pipeline)
                if len(executable_sequence) > 0:
                    ds = executable_sequence[-1].cache
                    if isinstance(ds, Dataset):
                        ds = ds.materialize()
        elif engine_name == 'spark':
            print("init spark")
            if not hasattr(self, 'rdp') or self.rdp is None:
                self.rdp = SparkDataProcessor()

            # execute
            with Timer(f"execute with spark"):
                for op in executable_sequence:
                    if ds != None and isinstance(op, DatasetReader):
                        op.cache = ds
                    else:
                        op.execute_spark(executable_pipeline, self.rdp)
                if len(executable_sequence) > 0:
                    ds = executable_sequence[-1].cache
                    if isinstance(ds, DataFrame):
                        ds = ds.cache()
                        total_len = ds.count()
                        
        self.engine_name = engine_name
        
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
           
class ResumableTextPipeline(TextPipeline):
    # Provide a pipeline for large dir. We will handle files one by one and resume when pipeline broken.
    def __init__(self, pipeline_file=None):
        super().__init__(pipeline_file)
            
    def execute(self):
        # Fix pipeline
        output_dir = ""
        for idx, op in self.pipeline.items():
            if op.op in ['SourcedParquetReader', 'ParquetReader']:
                op.op = 'PerfileSourcedParquetReader'
            if op.op in ['SourcedJsonlReader', 'JsonlReader']:
                op.op = 'PerfileSourcedJsonlReader'
            if op.op in ['ParquetWriter', 'PerfileParquetWriter']:
                op.op = 'PerfileParquetWriter'
                output_dir = op.config['output_dir']
        if output_dir == "":
            output_dir = f"ResumableTextPipeline_output_{time.strftime('%Y%m%d%H%M%S')}"
            self.add_operation(PerfileParquetWriter(output_dir = output_dir))
        
        # prepare output dir and record system
        os.makedirs(output_dir, exist_ok=True)
        status_log_path = os.path.join(output_dir, 'status.log')
        if os.path.exists(status_log_path):
            status_tracker = open(status_log_path, 'r+')
            done_files = [line.split(',')[0] for line in status_tracker.readlines()]
            status_tracker.write('\n')
        else:
            status_tracker = open(status_log_path, 'w')
            done_files = []
        self.export(os.path.join(output_dir, "pipeline.json"))
        self.plot()
        
        # prepare pipeline
        if not hasattr(self, 'executable_pipeline') or not hasattr(self, 'executable_sequence'):
            self.executable_pipeline, self.executable_sequence = self.create_executable_pipeline()
        executable_pipeline = self.executable_pipeline
        executable_sequence = self.executable_sequence
        
        engine_name = self.check_platform(executable_sequence)
        
        if engine_name != 'ray':
            raise NotImplementedError("ResumableTextPipeline only support operations with ray mode")
        self.engine_name = engine_name
        if not ray.is_initialized():
            print(f"init ray with total mem of {total_mem}")
            try:
                ray.init(object_store_memory=total_mem, num_cpus=total_cores)
            except:
                ray.init()

        # explode one pipeline to multiple sub pipeline (per file)
        sub_pipelines = []
        op_chain = []
        for op in executable_sequence:
            if isinstance(op, PerfileReader):
                op.execute_ray(executable_pipeline)
                sub_pipelines = op.cache
            elif len(sub_pipelines) > 0:
                op_chain.append(op)
                
        for ds_reader, source_id in (pbar := tqdm(sub_pipelines, total=len(sub_pipelines))):
            # check if we should skip
            if source_id in done_files:
                print(f"skip {source_id}, it was processed in last round")
                del ds_reader
                continue
            
            # If not skip, then
            pbar.set_description(f"ResumableTextPipeline, current on {source_id}")
            start = time.time()
            for idx, op in enumerate(op_chain):
                if idx == 0:
                    op.execute_ray(executable_pipeline, ds_reader)
                elif isinstance(op, PerfileParquetWriter):
                    op.execute_ray(executable_pipeline, source_id)
                else:
                    op.execute_ray(executable_pipeline)
            elapse = time.time() - start
            status_tracker.write(f"{source_id}, {elapse} secs\n")
            status_tracker.flush()
            done_files.append(status_tracker)
            del ds_reader
            
        print(f"Completed!\nResumableTextPipeline will not return dataset, please check {output_dir} for verification.")
        
        # fetch result
        return None
  
    