import json
import pandas as pd
from .dataframe import *
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD as SparkRDD
from pyrecdp.core.utils import dump_fix
from IPython.display import display

class Operation:
    def __init__(self, idx, children, output, op, config):
        self.idx = idx
        self.children = children #input operation
        self.output = output #output schema
        self.op = op #func name
        self.config = config #operation config

    def __repr__(self):
        return repr(self.dump())
    
    def dump(self):
        dump_dict = {
            #'idx': self.idx,
            'children': self.children,
            #'output': self.output,
            'op': self.op,
            'config': dump_fix(self.config)
        }
        return dump_dict
    
    def instantiate(self):
        from .data import DataFrameOperation, DataLoader
        from .merge import MergeOperation
        from .name import RenameOperation
        from .category import CategorifyOperation
        from .drop import DropOperation
        from .fillna import FillNaOperation
        from .featuretools_adaptor import FeaturetoolsOperation
        from .geograph import HaversineOperation
        from .type import TypeInferOperation
        from .tuple import TupleOperation
        from .custom import CustomOperation
        from .encode import OnehotEncodeOperation, ListOnehotEncodeOperation, TargetEncodeOperation
        from pyrecdp.primitives.estimators.lightgbm import LightGBM

        operations_ = {
            'DataFrame': DataFrameOperation,
            'DataLoader': DataLoader,
            'merge': MergeOperation,
            'rename': RenameOperation,
            'categorify': CategorifyOperation,
            'drop': DropOperation,
            'fillna': FillNaOperation,
            'haversine': HaversineOperation,
            'tuple': TupleOperation,
            'type_infer': TypeInferOperation,
            'lightgbm': LightGBM,
            'onehot_encode': OnehotEncodeOperation,
            'list_onehot_encode': ListOnehotEncodeOperation,
            'target_encode': TargetEncodeOperation,
            'custom_operator': CustomOperation
        }

        if self.op in operations_:
            return operations_[self.op](self)
        else:
            try:
                return FeaturetoolsOperation(self)
            except:
                raise NotImplementedError(f"operation {self.op} is not implemented.")
 
    @staticmethod
    def load(idx, dump_dict):
        obj = Operation(idx, dump_dict['children'], None, dump_dict['op'], dump_dict['config'])
        return obj

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
        self.fast_without_dpp = False
       
    def __repr__(self) -> str:
        return self.op.op

    def describe(self) -> str:
        return str(self.op.dump())
        
    def execute_pd(self, pipeline):
        _proc = self.get_function_pd()
        if not self.op.children or len(self.op.children) == 0:
            pass
        else:
            child_output = pipeline[self.op.children[0]].cache
            self.cache = _proc(child_output)
            
    def execute_spark(self, pipeline, rdp):
        _convert = None
        if not self.op.children or len(self.op.children) == 0:
            pass
        else:
            child_output = pipeline[self.op.children[0]].cache
            if isinstance(child_output, SparkDataFrame):
                if self.support_spark_dataframe:
                    _proc = self.get_function_spark(rdp)
                elif self.support_spark_rdd:
                    _convert = SparkDataFrameToRDDConverter().get_function(rdp)
                    _proc = self.get_function_spark_rdd(rdp)
                else:
                    _convert = SparkDataFrameToDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_pd()
            elif isinstance(child_output, SparkRDD):
                if self.support_spark_rdd:
                    _proc = self.get_function_spark_rdd(rdp)
                elif self.support_spark_dataframe:
                    _convert = RDDToSparkDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_spark(rdp)
                else:
                    _convert = RDDToDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_pd()
            elif isinstance(child_output, pd.DataFrame):
                if self.fast_without_dpp:
                    _proc = self.get_function_pd()
                elif self.support_spark_rdd:
                    _convert = DataFrameToRDDConverter().get_function(rdp)
                    _proc = self.get_function_spark_rdd(rdp)
                elif self.support_spark_dataframe:
                    _convert = DataFrameToSparkDataFrameConverter().get_function(rdp)
                    _proc = self.get_function_spark(rdp)
                else:
                    _proc = self.get_function_pd()
            else:
                raise ValueError(f"child cache is not recognized {child_output}")
        
            if _convert:
                child_output = _convert(child_output)
                pipeline[self.op.children[0]].cache = child_output
            self.cache = _proc(child_output)
            #print(self.cache.take(1))

    def get_function_spark_rdd(self, rdp):
        actual_func = self.get_function_pd()
        def transform(iter, *args):
            for x in iter:
                yield actual_func(x, *args)
        def base_spark_feature_generator(rdd):
            return rdd.mapPartitions(transform)
        return base_spark_feature_generator