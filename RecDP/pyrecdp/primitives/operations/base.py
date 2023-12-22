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

import json
from dataclasses import dataclass
from functools import wraps

import pandas as pd
from ray.data import Dataset

from .dataframe import *
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD as SparkRDD
from pyrecdp.core.utils import dump_fix
from IPython.display import display
from pyrecdp.core.registry import Registry
from pyrecdp.core.import_utils import check_availability_and_install

AUTOFEOPERATORS = Registry('BaseAutoFEOperation')
class Operation:
    def __init__(self, idx, children, output, op, config):
        self.idx = idx
        self.children = children  # input operation
        self.output = output  # output schema
        self.op = op  # func name
        self.config = config  # operation config

    def __repr__(self):
        if hasattr(self, 'dump_dict'):
            return repr(self.dump_dict)
        from copy import deepcopy
        dpcpy_obj = deepcopy(self)
        return repr(dpcpy_obj.dump())

    def dump(self, base_dir = ''):
        dump_dict = {
            # 'idx': self.idx,
            'children': self.children,
            # 'output': self.output,
            'op': self.op,
            'config': dump_fix(self.config, base_dir)
        }
        self.dump_dict = dump_dict
        return dump_dict

    def instantiate(self):
        if self.op in AUTOFEOPERATORS.modules:
            return AUTOFEOPERATORS.modules[self.op](self)
        elif self.op in LLMOPERATORS.modules:
            return LLMOPERATORS.modules[self.op].instantiate(self, self.config)
        else:
            #print(self.op)
            #print(AUTOFEOPERATORS.modules.keys())
            try:
                from pyrecdp.primitives.operations.featuretools_adaptor import FeaturetoolsOperation
                return FeaturetoolsOperation(self)
            except:
                raise NotImplementedError(f"operation {self.op} is not implemented.")

    @staticmethod
    def load(idx, dump_dict):
        obj = Operation(idx, dump_dict['children'], None, dump_dict['op'], dump_dict['config'])
        return obj


BASEOPERATORS = Registry('BaseOperation')


class BaseOperation:
    def __init__(self, op_base):
        # option1: for get_function_pd use
        if not isinstance(op_base, Operation):
            op_base = Operation(-1, None, [], f'{self.__class__.__name__}', op_base)
        # option2: complete usage in recdp
        self.op = op_base
        self.cache = None
        self.support_spark_dataframe = False
        self.support_spark_rdd = False
        self.support_spark_ray = False
        self.fast_without_dpp = False

    def __repr__(self) -> str:
        return self.op.op

    def describe(self) -> str:
        return str(self.op.dump())

    def execute_pd(self, pipeline, trans_type='fit_transform'):
        _proc = self.get_function_pd(trans_type)
        if not self.op.children or len(self.op.children) == 0:
            pass
        else:
            child_output = pipeline[self.op.children[0]].cache
            self.cache = _proc(child_output)

    def execute_spark(self, pipeline, rdp, trans_type='fit_transform'):
        _convert = None
        if not self.op.children or len(self.op.children) == 0:
            pass
        else:
            child_output = pipeline[self.op.children[0]].cache
            if isinstance(child_output, SparkDataFrame):
                if self.support_spark_dataframe:
                    _proc = self.get_function_spark(rdp, trans_type)
                elif self.support_spark_rdd:
                    _convert = SparkDataFrameToRDDConverter().get_function(rdp)
                    _proc = self.get_function_spark_rdd(rdp, trans_type)
                else:
                    _convert = SparkDataFrameToDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_pd(trans_type)
            elif isinstance(child_output, SparkRDD):
                if self.support_spark_rdd:
                    _proc = self.get_function_spark_rdd(rdp, trans_type)
                elif self.support_spark_dataframe:
                    _convert = RDDToSparkDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_spark(rdp, trans_type)
                else:
                    _convert = RDDToDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_pd(trans_type)
            elif isinstance(child_output, pd.DataFrame):
                if self.fast_without_dpp:
                    _proc = self.get_function_pd(trans_type)
                elif self.support_spark_rdd:
                    _convert = DataFrameToRDDConverter().get_function(rdp)
                    _proc = self.get_function_spark_rdd(rdp, trans_type)
                elif self.support_spark_dataframe:
                    _convert = DataFrameToSparkDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_spark(rdp, trans_type)
                else:
                    _proc = self.get_function_pd(trans_type)
            else:
                raise ValueError(f"child cache is not recognized {child_output}")

            if _convert:
                child_output = _convert(child_output)
                pipeline[self.op.children[0]].cache = child_output
            self.cache = _proc(child_output)
            # print(self.cache.take(1))

    def get_function_spark_rdd(self, rdp, trans_type='fit_transform'):
        actual_func = self.get_function_pd(trans_type)

        def transform(iter, *args):
            for x in iter:
                yield actual_func(x, *args)

        def base_spark_feature_generator(rdd):
            return rdd.mapPartitions(transform)

        return base_spark_feature_generator


LLMOPERATORS = Registry('BaseLLMOperation')


class BaseLLMOperation(BaseOperation):
    def __init__(self, args_dict={}, requirements=[]):
        self.op = Operation(-1, None, [], f'{self.__class__.__name__}', args_dict)
        self.cache = None
        self.support_spark = False
        self.support_ray = True
        self.statistics = OperationStatistics(0, 0, 0, 0)
        self.statistics_flag = False
        check_availability_and_install(requirements)

    @classmethod
    def instantiate(cls, op_obj, config):
        ins = cls(**config)
        ins.op = op_obj
        return ins

    def execute_ray(self, pipeline, child_ds=None):
        child_output = []
        if child_ds is not None:
            self.cache = self.process_rayds(child_ds)
        else:
            children = self.op.children if self.op.children is not None else []
            for op in children:
                child_output.append(pipeline[op].cache)
            self.cache = self.process_rayds(*child_output)
        return self.cache

    def execute_spark(self, pipeline, rdp, child_ds=None):
        child_output = []
        skip_first = False
        if child_ds is not None:
            child_output.append(child_ds)
            skip_first = True
        children = self.op.children if self.op.children is not None else []
        for idx, op in enumerate(children):
            if idx == 0 and skip_first:
                continue
            child_output.append(pipeline[op].cache)
        print(self)
        self.cache = self.process_spark(rdp.spark, *child_output)
        return self.cache

    def process_rayds(self, ds=None):
        return self.cache

    def process_spark(self, spark, df=None):
        return self.cache

    def process_row(self, sample: dict, text_key, new_name, actual_func, *actual_func_args) -> dict:
        sample[new_name] = actual_func(sample[text_key], *actual_func_args)
        return sample

    def process_batch(self, sample: dict) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} does not support process_batch yet")

    def get_modified_rows(self):
        self.statistics.total_changed = self.statistics.total_in - self.statistics.total_out

    def summarize(self) -> str:
        self.get_modified_rows()
        return (
            f"A total of {self.statistics.total_in} rows of data were processed, using {self.statistics.used_time} seconds, "
            f"with {self.statistics.total_changed} rows modified or removed, {self.statistics.total_out} rows of data remaining.")

class DummyOperation(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.support_spark_dataframe = True
        self.support_spark_rdd = True

    def get_function_pd(self, trans_type='fit_transform'):
        def dummy_op(df):
            return df

        return dummy_op

    def get_function_spark(self, rdp, trans_type='fit_transform'):
        def dummy_op(df):
            return df

        return dummy_op
AUTOFEOPERATORS.register(DummyOperation, "time_series_infer")


@dataclass
class OperationStatistics:
    total_in: int
    total_out: int
    total_changed: int
    used_time: float


def statistics_decorator(func):
    def wrapper(self, *args, **kwargs):
        if self.statistics_flag:
            if isinstance(args[0], Dataset):
                ds = args[0]
                self.statistics.total_in += ds.count()
                start = time.time()
                result = func(self, *args, **kwargs)
                self.statistics.total_out += result.count()
                elapse = time.time() - start
                self.statistics.used_time += elapse
            elif isinstance(args[1], SparkDataFrame):
                print("statistics_decorator spark")
                df = args[1]
                self.statistics.total_in += df.count()
                start = time.time()
                result = func(self, *args, **kwargs)
                self.statistics.total_out += result.cache().count()
                elapse = time.time() - start
                self.statistics.used_time += elapse

        else:
            result = func(self, *args, **kwargs)
        return result

    return wrapper
