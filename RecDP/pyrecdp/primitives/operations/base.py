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


class Operation:
    def __init__(self, idx, children, output, op, config):
        self.idx = idx
        self.children = children  # input operation
        self.output = output  # output schema
        self.op = op  # func name
        self.config = config  # operation config

    def __repr__(self):
        return repr(self.dump())

    def dump(self):
        dump_dict = {
            # 'idx': self.idx,
            'children': self.children,
            # 'output': self.output,
            'op': self.op,
            'config': dump_fix(self.config)
        }
        return dump_dict

    def instantiate(self):
        from .data import DataFrameOperation, DataLoader
        from .merge import MergeOperation
        from .name import RenameOperation
        from .category import CategorifyOperation, GroupCategorifyOperation
        from .drop import DropOperation
        from .fillna import FillNaOperation
        from .featuretools_adaptor import FeaturetoolsOperation
        from .geograph import HaversineOperation
        from .type import TypeInferOperation
        from .tuple import TupleOperation
        from .custom import CustomOperation
        from .encode import OnehotEncodeOperation, ListOnehotEncodeOperation, TargetEncodeOperation, \
            CountEncodeOperation
        from pyrecdp.primitives.estimators.lightgbm import LightGBM

        operations_ = {
            'DataFrame': DataFrameOperation,
            'DataLoader': DataLoader,
            'merge': MergeOperation,
            'rename': RenameOperation,
            'categorify': CategorifyOperation,
            'group_categorify': GroupCategorifyOperation,
            'drop': DropOperation,
            'fillna': FillNaOperation,
            'haversine': HaversineOperation,
            'tuple': TupleOperation,
            'type_infer': TypeInferOperation,
            'lightgbm': LightGBM,
            'onehot_encode': OnehotEncodeOperation,
            'list_onehot_encode': ListOnehotEncodeOperation,
            'target_encode': TargetEncodeOperation,
            'count_encode': CountEncodeOperation,
            'custom_operator': CustomOperation,
            'time_series_infer': DummyOperation,
        }

        if self.op in operations_:
            return operations_[self.op](self)
        elif self.op in LLMOPERATORS.modules:
            return LLMOPERATORS.modules[self.op].instantiate(self, self.config)
        else:
            try:
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
    def __init__(self, args_dict={}):
        self.op = Operation(-1, None, [], f'{self.__class__.__name__}', args_dict)
        self.cache = None
        self.support_spark = False
        self.support_ray = True
        self.statistics = OperationStatistics(0, 0, 0, 0)
        self.statistics_flag = False

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

    def union_ray_ds(self, ds1, ds2):
        def add_new_empty_column(content, column_name):
            content[column_name] = None
            return content
        def convert_to_string(content, column_name):
            content[column_name] = str(content[column_name])
            return content
        for column in [column for column in ds2.columns() if column not in ds1.columns()]:
            ds1 = ds1.map(lambda x: add_new_empty_column(x, column))
        for column in [column for column in ds1.columns() if column not in ds2.columns()]:
            ds2 = ds2.map(lambda x: add_new_empty_column(x, column))
        ds1_fields_dict  =dict(zip(ds1.schema().names, ds1.schema().types))
        ds2_fields_dict = dict(zip(ds2.schema().names, ds2.schema().types))
        for column_name in ds1_fields_dict.keys():
            if ds2_fields_dict[column_name] != ds1_fields_dict[column_name] and not (
                    str(ds2_fields_dict[column_name]) == "null" or str(ds1_fields_dict[column_name]) == "null"):
                ds1 = ds1.map(lambda x: convert_to_string(x, column_name))
                ds2 = ds2.map(lambda x: convert_to_string(x, column_name))
        return ds1.union(ds2)

    def union_spark_df(self, df1, df2):
        from pyspark.sql.functions import lit
        import pyspark.sql.functions as F
        from pyspark.sql.types import NullType, StringType
        for column in [column for column in df2.columns if column not in df1.columns]:
            df1 = df1.withColumn(column, lit(None))
        for column in [column for column in df1.columns if column not in df2.columns]:
            df2 = df2.withColumn(column, lit(None))
        df1_fields, df2_fields = df1.schema.fields, df2.schema.fields
        df1_fields_dict, df2_fields_dict = {}, {}
        for df1_field in df1_fields:
            df1_fields_dict[df1_field.name] = df1_field.dataType
        for df2_field in df2_fields:
            df2_fields_dict[df2_field.name] = df2_field.dataType
        for column_name in df1_fields_dict.keys():
            if df2_fields_dict[column_name] != df1_fields_dict[column_name] and not (
                    df2_fields_dict[column_name] == NullType() or df1_fields_dict[column_name] == NullType()):
                df1 = df1.withColumn(column_name, F.col(column_name).cast(StringType()))
                df2 = df2.withColumn(column_name, F.col(column_name).cast(StringType()))
        return df1.union(df2)


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
