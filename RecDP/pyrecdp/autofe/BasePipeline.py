from pyrecdp.primitives.generators import *
from pyrecdp.core import DataFrameSchema, SparkDataProcessor, DiGraph
from pyrecdp.primitives.operations import Operation, DataFrameOperation, RDDToDataFrameConverter
import pandas as pd
import logging
import graphviz
import json
from pyrecdp.core.utils import Timer, sample_read
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD as SparkRDD

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class BasePipeline:
    def __init__(self, dataset, label, *args, **kwargs):
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
        if not main_table:
            raise ValueError(f"label {label} is not found in dataset")
        
        # Set properties for BasePipeline
        if not input_is_path:
            original_data = self.dataset[main_table]
        else:
            original_data = sample_read(self.dataset[main_table])
        y = original_data[label]
        to_select = [i for i in original_data.columns if i != y.name]
        self.feature_data = original_data[to_select]
            
        self.generators = []
        self.pipeline = DiGraph()
        if not input_is_path:
            self.pipeline[0] = Operation(
                0, None, output = DataFrameSchema(self.feature_data), op = 'DataFrame', config = main_table)
        else:
            self.pipeline[0] = Operation(
                0, None, output = DataFrameSchema(self.feature_data), op = 'DataLoader', config = {'table_name': main_table, 'file_path': self.dataset[main_table]})

        if len(self.dataset) > 1:
            self.supplementary = dict((k, v) for k, v in self.dataset.items() if k != main_table)
        else:
            self.supplementary = None
        self.rdp = None
    
    def fit_analyze(self, *args, **kwargs):
        child = list(self.pipeline.keys())[-1]
        max_id = child
        for i in range(len(self.generators)):
            for generator in self.generators[i]:
                self.pipeline, child, max_id = generator.fit_prepare(self.pipeline, [child], max_id)

    def __repr__(self):
        return repr(self.pipeline)

    def export(self, file_path = None):
        json_object = self.pipeline.json_dump()
        if file_path:
            # Writing to sample.json
            with open("file_path", "w") as outfile:
                outfile.write(json_object)
        else:
            print(json_object)
                
    def plot(self):
        f = graphviz.Digraph()
        edges = []
        nodes = []
        f.attr(fontsize='10')
        def add_escape(input):
            input = input.replace('<', '\<').replace('>', '\>')
            #input = input.replace("'", "\\\'").replace("\"", "\\\"")
            return input

        def add_break(input):
            if isinstance(input, list) and len(input) < 3:
                for line in input:
                    if isinstance(line, str):
                        ret = str(input)
                        return ret
            if isinstance(input, dict):
                input = [f"{k}: {add_break(v)}" for k, v in input.items()]
            if isinstance(input, list):
                ret = ""
                for line in input:
                    ret += str(add_break(line)) + "\l"
                return ret
            return input

        for node_id, config in self.pipeline.items():
            nodes.append([str(node_id), f"{config.op} |{add_escape(str(add_break(config.config)))}"])
            if config.children:
                for src_id in config.children:
                    edges.append([str(src_id), str(node_id)])
        for node in nodes:
            f.node(node[0], node[1], shape='record', fontsize='12')
        for edge in edges:
            f.edge(*edge)
        return f  

    def to_chain(self):
        return self.pipeline.convert_to_node_chain()
        
    def execute(self, engine_type = "pandas"):
        # prepare pipeline
        node_chain = self.pipeline.convert_to_node_chain()
        executable_pipeline = DiGraph()
        executable_sequence = []
        for idx in node_chain:
            executable_pipeline[idx] = self.pipeline[idx].instantiate()
            executable_sequence.append(executable_pipeline[idx])

        # execute
        if engine_type == 'pandas':
            with Timer(f"execute with pandas"):
                for op in executable_sequence:
                    if isinstance(op, DataFrameOperation):
                        op.set(self.dataset)
                    with Timer(f"execute {op}"):
                        op.execute_pd(executable_pipeline)
            df = executable_sequence[-1].cache
        elif engine_type == 'spark':
            for op in executable_sequence:
                if isinstance(op, DataFrameOperation):
                    op.set(self.dataset)
                print(f"append {op}")
                op.execute_spark(executable_pipeline, self.rdp)
            ret = executable_sequence[-1].cache
            if isinstance(ret, SparkDataFrame):
                with Timer(f"execute with spark"):
                    df = self.rdp.transform(ret)
            elif isinstance(ret, SparkRDD):
                _convert = RDDToDataFrameConverter().get_function(self.rdp)
                with Timer(f"execute with spark"):
                    df = _convert(ret)
            else:
                raise ValueError(f"unrecognized {ret} produced by execute with spark")
        else:
            raise NotImplementedError('pipeline only support pandas and spark as engine')
        
        # fetch result
        return df

    def fit_transform(self, engine_type = 'pandas', *args, **kwargs):
        if engine_type == "spark":
            self.rdp = SparkDataProcessor()
        ret = self.execute(engine_type)
        if engine_type == "spark":
            del self.rdp 
        return ret