from pyrecdp.primitives.generators import *
from pyrecdp.core.schema import DataFrameSchema
from pyrecdp.core.di_graph import DiGraph
from pyrecdp.data_processor import DataProcessor as SparkDataProcessor
from pyrecdp.core.pipeline import BasePipeline
from pyrecdp.primitives.operations import Operation, DataFrameOperation, RDDToDataFrameConverter, SparkDataFrameToDataFrameConverter, TargetEncodeOperation
import pandas as pd
import logging
import json
from pyrecdp.core.utils import Timer, sample_read, deepcopy
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD as SparkRDD
from IPython.display import display
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class TabularPipeline(BasePipeline):
    def __init__(self, dataset, label, exclude_op = [], include_op = [], *args, **kwargs):
        # properties
        # self.main_table: main table names
        # self.dataset: a dictionary, main_table will be indicated with key 'main_table'
        # self.y: target label
        # self.pipeline: a direct graph to store the operation
        super().__init__()
        if isinstance(dataset, pd.DataFrame):
            self.dataset = {'main_table': dataset}
        elif isinstance(dataset, list):
            self.dataset = dict((idx, data) for idx, data in enumerate(dataset))
        elif isinstance(dataset, dict):
            self.dataset = dataset
        else:
            self.dataset = {'main_table': dataset}
        main_table = None
        input_is_path = False
        if isinstance(label, str):    
            for data_key, data in self.dataset.items():
                # optional: data is file_path
                if isinstance(data, str):
                    input_is_path = True
                    data = sample_read(data)
                if label in data.columns:
                    main_table = data_key
                    break
            self.y = label
        elif isinstance(label, pd.Series):
            self.y = label.name
        else:
            self.y = None
        if not main_table:
             main_table = 'main_table'
        if not main_table:
            raise ValueError(f"label {label} is not found in dataset")
        self.main_table = main_table
        
        # Set properties for BasePipeline
        if not input_is_path:
            original_data = self.dataset[main_table]
        else:
            original_data = sample_read(self.dataset[main_table])
        
        self.feature_columns = [i for i in original_data.columns if i != self.y]
        #feature_data = original_data[self.feature_columns]
        self.exclude_op = exclude_op
        self.include_op = include_op
        self.generators = []
        if not input_is_path:
            op = 'DataFrame'
            config = main_table
        else:
            op = 'DataLoader'
            config = {'table_name': main_table, 'file_path': self.dataset[main_table]}
        
        cur_id = 0
        self.pipeline[cur_id] = Operation(
            cur_id, None, output = DataFrameSchema(original_data), op = op, config = config)

        if len(self.dataset) > 1:
            self.supplementary = dict((k, v) for k, v in self.dataset.items() if k != main_table)
        else:
            self.supplementary = None

    def update_label(self):
        leaf_idx = self.pipeline.convert_to_node_chain()[-1]
        pa_schema = self.pipeline[leaf_idx].output
        label_list = [pa_field.name for pa_field in pa_schema if pa_field.is_label]
        if len(label_list) > 0:
            self.y = label_list[0]
        else:
            self.y = None
     
    def fit_analyze(self, *args, **kwargs):
        child = list(self.pipeline.keys())[-1]
        max_id = child
        to_run = []
        for i in range(len(self.generators)):
            for generator in self.generators[i]:
                if generator.__class__.__name__ in self.exclude_op:
                    continue
                to_run.append(generator)
        
        pbar = tqdm(to_run, total=len(to_run))
        for generator in pbar:
            pbar.set_description(f"{generator.__class__.__name__}")
            self.pipeline, child, max_id = generator.fit_prepare(self.pipeline, [child], max_id)
        return child, max_id
    
    def add_operation(self, config):
        # check if below keys are existing
        # config can be either a dict or function defininition
        pipeline_chain = self.to_chain()
        if not hasattr(self, 'transformed_end_idx'):            
            leaf_child = pipeline_chain[-1]
        else:
            leaf_child = self.pipeline[self.transformed_end_idx].children[0]
        
        if not isinstance(config, dict):
            op = config
            config = {
                "children": [leaf_child],
                "inline_function": op,
            }        
        children = config["children"]
        inline_function = config["inline_function"]
        
        if not isinstance(children, list):
            children = [children]
        
        # get current max operator id
        max_idx = self.pipeline.get_max_idx()
        cur_idx = max_idx + 1
        
        config = {
            "func_name": inline_function,
        }
        self.pipeline[cur_idx] = Operation(
            cur_idx, children, output = None, op = "custom_operator", config = config)
        
        # we need to find nexts
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
                    
    def execute(self, engine_type = "pandas", start_op_idx = -1, no_cache = False, transformed_end = -1, data = None, trans_type = 'fit_transform'):
        # prepare pipeline
        if not hasattr(self, 'executable_pipeline') or not hasattr(self, 'executable_sequence'):
            self.executable_pipeline, self.executable_sequence = self.create_executable_pipeline()
        executable_pipeline = self.executable_pipeline
        executable_sequence = self.executable_sequence

        # execute
        if engine_type == 'pandas':
            with Timer(f"execute with pandas"):
                start = False
                for op in executable_sequence:
                    if start_op_idx == -1 or op.op.idx == start_op_idx:
                        start = True
                    if not start:
                        continue
                    if isinstance(op, DataFrameOperation):
                        if data:
                            input_df = data
                        else:
                            input_df = self.dataset if start_op_idx == -1 else {'main_table': self.transformed_cache}
                        input_df = deepcopy(input_df) if no_cache else input_df
                        op.set(input_df)
                    with Timer(f"execute {op}"):
                        op.execute_pd(executable_pipeline, trans_type = trans_type)
            if transformed_end == -1:
                df = executable_sequence[-1].cache
            else:
                df = executable_pipeline[transformed_end].cache
        elif engine_type == 'spark':
            with Timer(f"execute with spark"):
                start = False
                for op in executable_sequence:
                    if start_op_idx == -1 or op.op.idx == start_op_idx:
                        start = True
                    if not start:
                        continue
                    if isinstance(op, DataFrameOperation):
                        if data:
                            input_df = data
                        else:
                            input_df = self.dataset if start_op_idx == -1 else {'main_table': self.transformed_cache}
                        input_df = deepcopy(input_df) if no_cache else input_df
                        op.set(input_df)
                    print(f"append {op}")
                    op.execute_spark(executable_pipeline, self.rdp, trans_type = trans_type)
                if transformed_end == -1:
                    ret = executable_sequence[-1].cache
                else:
                    ret = executable_pipeline[transformed_end].cache
                if isinstance(ret, SparkDataFrame):
                    _convert = SparkDataFrameToDataFrameConverter().get_function(self.rdp)
                    df = _convert(ret)
                elif isinstance(ret, SparkRDD):
                    _convert = RDDToDataFrameConverter().get_function(self.rdp)
                    df = _convert(ret)
                else:
                    df = ret
        else:
            raise NotImplementedError('pipeline only support pandas and spark as engine')
        
        # fetch result
        return df

    def fit_transform(self, engine_type = 'pandas', no_cache = False, data = None, *args, **kwargs):
        if not no_cache and hasattr(self, 'transformed_cache') and self.transformed_cache is not None:
            print("Detect pre-transformed cache, return cached data")
            print("If re-transform is required, please use fit_transform(no_cache = True)")
            return self.transformed_cache
        if engine_type == "spark":
            if "spark_master" in kwargs:
                spark_master = kwargs["spark_master"]
                spark_mode = 'standalone'
            else:
                spark_master = "local[*]"
                spark_mode = 'local'
            self.rdp = SparkDataProcessor(spark_mode=spark_mode, spark_master=spark_master)
        ret = self.execute(engine_type = engine_type, no_cache = no_cache, data = data)
        if engine_type == "spark":
            del self.rdp 
            self.rdp = None
        self.transformed_cache = ret
        return ret
    
    def transform(self, engine_type = 'pandas', no_cache = False, data = None, *args, **kwargs):
        if not no_cache and hasattr(self, 'transformed_cache') and self.transformed_cache is not None:
            print("Detect pre-transformed cache, return cached data")
            print("If re-transform is required, please use fit_transform(no_cache = True)")
            return self.transformed_cache
        if engine_type == "spark":
            if "spark_master" in kwargs:
                spark_master = kwargs["spark_master"]
                spark_mode = 'standalone'
            else:
                spark_master = "local[*]"
                spark_mode = 'local'
            self.rdp = SparkDataProcessor(spark_mode=spark_mode, spark_master=spark_master)
        ret = self.execute(engine_type = engine_type, no_cache = no_cache, data = data, trans_type = 'transform')
        self.transformed_cache = ret
        if engine_type == "spark":
            del self.rdp
            self.rdp = None
        return ret