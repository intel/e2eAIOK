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

from pyrecdp.core.di_graph import DiGraph
from pyrecdp.core.pipeline import BasePipeline
from pyrecdp.primitives.operations import Operation, BaseOperation
from pyrecdp.primitives.operations.text_reader import DatasetReader, PerfileReader
from pyrecdp.primitives.operations.text_writer import PerfileParquetWriter, PerfileJsonlWriter, ParquetWriter
import logging
from pyrecdp.core.utils import Timer, deepcopy
from IPython.display import display
from tqdm import tqdm
import types
from ray.data import Dataset
from pyspark.sql import DataFrame
import ray
from pyrecdp.data_processor import DataProcessor as SparkDataProcessor
import time
import os
import psutil
from pyrecdp.primitives.operations.logging_utils import logger
import json

total_mem = int(psutil.virtual_memory().total * 0.6)
total_cores = psutil.cpu_count(logical=False)


class TextPipeline(BasePipeline):
    def __init__(self, engine_name='ray', pipeline_file=None):
        super().__init__()
        self.engine_name = engine_name
        self.ray_start_by_us = False
        if pipeline_file != None:
            self.import_from_json(pipeline_file) if pipeline_file.endswith(
                '.json') else self.import_from_yaml(pipeline_file)
            self.plot()
        else:
            # add a data set input place holder
            op = DatasetReader()
            self.add_operation(op)

    def __del__(self):
        if self.ray_start_by_us:
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
            if self.engine_name == 'spark' and is_spark:
                return 'spark'
            else:
                return 'ray'
        elif is_spark:
            return 'spark'
        else:
            print(
                f"We can't identify an uniform engine for this pipeline. \n  Operations work on Ray are {ray_list}. \n  Operations work on Spark are {spark_list}")
            return 'mixed'

    def optimize_execute_plan(self):
        # Update Writer
        output_dir = ""
        for idx, op in self.pipeline.items():
            if op.op in ['ParquetWriter', 'PerfileParquetWriter']:
                output_dir = op.config['output_dir']
            if op.op in ['JsonlWriter', 'PerfileJsonlWriter']:
                output_dir = op.config['output_dir']
            if op.op in ['DocumentIngestion']:
                output_dir = f"TextPipeline_vectordatabase_{time.strftime('%Y%m%d%H%M%S')}"
        if output_dir == "":
            output_dir = f"TextPipeline_output_{time.strftime('%Y%m%d%H%M%S')}"
            self.add_operation(ParquetWriter(output_dir=output_dir))
        return output_dir
    
    def execute(self, ds=None):
        output_dir = self.optimize_execute_plan()
        os.makedirs(output_dir, exist_ok=True)
        self.export(os.path.join(output_dir, "pipeline.json"))
        self.plot(os.path.join(output_dir, "pipeline"))
        logger.add(os.path.join(output_dir, "pipeline.log"))

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
                self.ray_start_by_us = True

            # execute
            with Timer(f"execute with ray"):
                for op in executable_sequence:
                    if ds is not None and isinstance(op, DatasetReader):
                        if not isinstance(ds, Dataset):
                            ds = ray.data.from_pandas(ds)
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
                    if ds is not None and isinstance(op, DatasetReader):
                        if not isinstance(ds, DataFrame):
                            ds = self.rdp.spark.createDataFrame(ds)
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

    def add_operation(self, config, function_type = 'map', text_key = 'text'):
        # get current max operator id
        max_idx = self.pipeline.get_max_idx()
        cur_idx = max_idx + 1
        find_children_skip = False
        pipeline_chain = self.to_chain()

        if not isinstance(config, dict):
            op = config
            if max_idx == -1:
                leaf_child = None
            else:
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
                "func": inline_function,
                "text_key": text_key
            }
            if function_type == 'map':
                self.pipeline[cur_idx] = Operation(
                    cur_idx, children, output=None, op="TextCustomerMap", config=config)
            elif function_type == 'flatmap':
                self.pipeline[cur_idx] = Operation(
                    cur_idx, children, output=None, op="TextCustomerFlatMap", config=config)
            elif function_type == 'filter':
                self.pipeline[cur_idx] = Operation(
                    cur_idx, children, output=None, op="TextCustomerFilter", config=config)
            else:
                raise NotImplementedError(f"{function_type} is not supported as customer function yet")
        elif isinstance(inline_function, BaseOperation):
            op_name = inline_function.op.op
            # config = vars(inline_function)
            config = inline_function.op.config
            self.pipeline[cur_idx] = Operation(
                cur_idx, children, output=None, op=op_name, config=config)

        # we need to find nexts
        if find_children_skip:
            return self.pipeline
        # because we insert a new operation, replace next operator with new children
        for to_replace_child in children:
            # iterate all children, and find next operators
            next = []
            for idx in pipeline_chain:
                if self.pipeline[idx].children and to_replace_child == self.pipeline[idx].children[0]:
                    # we only replace next when its first child matches.
                    # This is a fix for deduplication, because we don't want to replace dictDF operator.
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
        return self.pipeline[cur_idx]

    def add_operations(self, config_list):
        for op in config_list:
            self.add_operation(op)
        return self.pipeline

    def profile(self):
        # TODO: print analysis and log for each component.
        pass

    def evaluate(self) -> dict:
        self.execute()
        import random
        return {"metric_1": random.uniform(0, 1), "metric_2": random.uniform(0, 1)}


class ResumableTextPipeline(TextPipeline):
    # Provide a pipeline for large dir. We will handle files one by one and resume when pipeline broken.
    def __init__(self, engine_name='ray', pipeline_file=None):
        super().__init__(engine_name, pipeline_file)
        # Enabling this option will result in a decrease in execution speed
        self.statistics_flag = False

    def enable_statistics(self):
        logger.warning("Enabling this option will result in a decrease in execution speed")
        self.statistics_flag = True

    def duplicate_reader(self):
        from copy import deepcopy
        op = self.find_operation(['SourcedParquetReader', 'ParquetReader', 'PerfileSourcedParquetReader', 'GlobalParquetReader'])
        if op is not None:
            from pyrecdp.primitives.operations import GlobalParquetReader
            reader_op = self.add_operation(config={"children": deepcopy(op.children), "inline_function": GlobalParquetReader(**op.config)})
            return reader_op

        op = self.find_operation(['SourcedJsonlReader', 'JsonlReader', 'PerfileSourcedJsonlReader', 'GlobalJsonlReader'])
        if op is not None:
            from pyrecdp.primitives.operations import GlobalJsonlReader
            reader_op = self.add_operation(config={"children": deepcopy(op.children), "inline_function": GlobalJsonlReader(**op.config)})
            return reader_op                

    def optimize_execute_plan(self):
        # update for deduplication
        op = self.find_operation(['FuzzyDeduplicate'])
        if op is not None:
            from pyrecdp.primitives.operations import FuzzyDeduplicateGenDict
            duplicated_reader_op = self.duplicate_reader()
            gendict_op = self.add_operation(config={"children": duplicated_reader_op.idx, "inline_function": FuzzyDeduplicateGenDict(**op.config)})
            op.op = 'FuzzyDeduplicateApplyDict'
            op.children.append(gendict_op.idx)
        
        op = self.find_operation(['GlobalDeduplicate'])
        if op is not None:
            from pyrecdp.primitives.operations import GlobalDeduplicateGenDict
            duplicated_reader_op = self.duplicate_reader()
            gendict_op = self.add_operation(config={"children": duplicated_reader_op.idx, "inline_function": GlobalDeduplicateGenDict(**op.config)})
            op.op = 'GlobalDeduplicateApplyDict'
            op.children.append(gendict_op.idx)
        
        def skip(children):
            for c in children:
                if self.pipeline[c].op in ['FuzzyDeduplicateGenDict', 'GlobalDeduplicateGenDict']:
                    return True
            return False
        # Update Writer
        output_dir = ""
        for idx, op in self.pipeline.items():
            if op.op in ['SourcedParquetReader', 'ParquetReader']:
                op.op = 'PerfileSourcedParquetReader'
            if op.op in ['SourcedJsonlReader', 'JsonlReader']:
                op.op = 'PerfileSourcedJsonlReader'
            if op.op in ['ParquetWriter', 'PerfileParquetWriter']:
                op.op = 'PerfileParquetWriter'
                output_dir = op.config['output_dir']
            if op.op in ['JsonlWriter', 'PerfileJsonlWriter']:
                op.op = 'PerfileJsonlWriter'
                output_dir = op.config['output_dir']
            if op.op in ['DocumentIngestion']:
                output_dir = f"ResumableTextPipeline_vectordatabase_{time.strftime('%Y%m%d%H%M%S')}"
        if output_dir == "":
            output_dir = f"ResumableTextPipeline_output_{time.strftime('%Y%m%d%H%M%S')}"
            self.add_operation(PerfileParquetWriter(output_dir=output_dir))

        return output_dir

    def op_summary(self, op, output_dir):
        summarize_result = op.summarize()
        if isinstance(summarize_result, tuple):
            logger.info(f"{op.__class__.__name__}: {summarize_result[1]}")
            with open(os.path.join(output_dir, f"{op.__class__.__name__}-statistics"), "w+") as fout:
                fout.write(json.dumps(summarize_result[0]))
        else:
            logger.info(f"{op.__class__.__name__}: {summarize_result}")

    def execute(self):
        # Fix pipeline
        output_dir = self.optimize_execute_plan()
        os.makedirs(output_dir, exist_ok=True)
        self.export(os.path.join(output_dir, "pipeline.json"))
        self.plot(os.path.join(output_dir, "pipeline"))
        logger.add(os.path.join(output_dir, "pipeline.log"))

        # prepare output dir and record system
        status_log_path = os.path.join(output_dir, 'status.log')
        if os.path.exists(status_log_path):
            status_tracker = open(status_log_path, 'r+')
            done_files = [line.split(',')[0] for line in status_tracker.readlines()]
            status_tracker.write('\n')
        else:
            status_tracker = open(status_log_path, 'w')
            done_files = []
        

        # prepare pipeline
        if not hasattr(self, 'executable_pipeline') or not hasattr(self, 'executable_sequence'):
            self.executable_pipeline, self.executable_sequence = self.create_executable_pipeline()
        executable_pipeline = self.executable_pipeline
        executable_sequence = self.executable_sequence

        print(executable_sequence)
        
        engine_name = self.check_platform(executable_sequence)
        self.engine_name = engine_name
        # explode one pipeline to multiple sub pipeline (per file)
        sub_pipelines = []
        global_data = None
        op_chain = []

        if engine_name == 'ray':
            if not ray.is_initialized():
                print(f"init ray with total mem of {total_mem}")
                try:
                    ray.init(object_store_memory=total_mem, num_cpus=total_cores)
                except:
                    ray.init()
                self.ray_start_by_us = True

            for op in executable_sequence:
                if isinstance(op, PerfileReader):
                    op.statistics_flag = self.statistics_flag
                    op.execute_ray(executable_pipeline)
                    sub_pipelines = op.cache
                elif len(sub_pipelines) > 0:
                    op.statistics_flag = self.statistics_flag
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
                    elif isinstance(op, PerfileParquetWriter) or isinstance(op, PerfileJsonlWriter):
                        op.execute_ray(executable_pipeline, source_id)
                    else:
                        op.execute_ray(executable_pipeline)
                elapse = time.time() - start
                status_tracker.write(f"{source_id}, {elapse} secs\n")
                status_tracker.flush()
                done_files.append(status_tracker)
                if self.statistics_flag:
                    for op in op_chain:
                        self.op_summary(op, output_dir)
                del ds_reader
        elif engine_name == 'spark':
            if not hasattr(self, 'rdp') or self.rdp is None:
                self.rdp = SparkDataProcessor()

            # To process every op before Perfile Reader
            with Timer(f"execute with spark for global tasks"):                
                for op in executable_sequence:
                    if not isinstance(op, PerfileReader):
                        op.statistics_flag = self.statistics_flag
                        op.execute_spark(executable_pipeline, self.rdp)
                        if self.statistics_flag:
                            op_chain.append(op)
                    else:
                        break
                if self.statistics_flag:
                    for op in op_chain:
                        self.op_summary(op, output_dir)
                    op_chain = []
            
            # To process since Perfile Reader
            for op in executable_sequence:
                if isinstance(op, PerfileReader):
                    op.statistics_flag = self.statistics_flag
                    op.execute_spark(executable_pipeline, rdp = self.rdp)
                    sub_pipelines = op.cache
                elif len(sub_pipelines) > 0:
                    op.statistics_flag = self.statistics_flag
                    op_chain.append(op)

            # execute
            for ds_reader, source_id in (pbar := tqdm(sub_pipelines, total=len(sub_pipelines))):
                # check if we should skip
                if source_id in done_files:
                    print(f"skip {source_id}, it was processed in last round")
                    del ds_reader
                    continue

                # If not skip, then
                pbar.set_description(f"ResumableTextPipeline, current on {source_id}")
                print(source_id)
                start = time.time()
                for idx, op in enumerate(op_chain):
                    if idx == 0:
                        op.execute_spark(executable_pipeline, rdp = self.rdp, child_ds = ds_reader)
                    elif isinstance(op, PerfileParquetWriter) or isinstance(op, PerfileJsonlWriter):
                        op.execute_spark(executable_pipeline, source_id = source_id)
                    else:
                        op.execute_spark(executable_pipeline, rdp=self.rdp)

                elapse = time.time() - start
                status_tracker.write(f"{source_id}, {elapse} secs\n")
                status_tracker.flush()
                done_files.append(status_tracker)
                if self.statistics_flag:
                    for op in op_chain:
                        self.op_summary(op, output_dir)
                            

                del ds_reader
        else:
            raise NotImplementedError(f"ResumableTextPipeline is not support {engine_name} yet")
        
        logger.info(
            f"Completed! ResumableTextPipeline will not return dataset, please check {output_dir} for verification.")
        
        status_tracker.close()

        # fetch result
        return None
