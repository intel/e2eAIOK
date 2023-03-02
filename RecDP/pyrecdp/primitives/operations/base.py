import json
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
            'config': self.config
        }
        return dump_dict
    
    def instantiate(self):
        if self.op == 'DataFrame':
            from .data import DataFrameOperation
            return DataFrameOperation(self)
        if self.op == 'DataLoader':
            from .data import DataLoader
            return DataLoader(self)
        if self.op == 'merge':
            from .merge import MergeOperation
            return MergeOperation(self)
        if self.op == 'rename':
            from .name import RenameOperation
            return RenameOperation(self)
        if self.op == 'categorify':
            from .category import CategorifyOperation
            return CategorifyOperation(self)
        if self.op == 'datetime_feature':
            from .datetime import DatetimeOperation
            return DatetimeOperation(self)
        if self.op == 'drop':
            from .drop import DropOperation
            return DropOperation(self)
        if self.op == 'fillna':
            from .fillna import FillNaOperation
            return FillNaOperation(self)
        if self.op == 'haversine':
            from .geograph import HaversineOperation
            return HaversineOperation(self)
        if self.op == 'coordinates_infer':
            from .geograph import CoordinatesOperation
            return CoordinatesOperation(self)
        if self.op == 'bert_decode':
            from .nlp import BertDecodeOperation
            return BertDecodeOperation(self)
        if self.op == 'text_feature':
            from .nlp import TextFeatureGenerator
            return TextFeatureGenerator(self)
        if self.op == 'type_infer':
            from .type import TypeInferOperation
            return TypeInferOperation(self)
        raise NotImplementedError(f"operation {self.op} is not implemented.")
 
    @staticmethod
    def load(dump_dict):
        obj = Operation(dump_dict['idx'], dump_dict['children'].copy(), dump_dict['output'].copy(), dump_dict['op'].copy(), dump_dict['config'].copy())
        return obj

class BaseOperation:
    def __init__(self, op_base):
        self.op = op_base
        self.cache = None
       
    def __repr__(self) -> str:
        return self.op.op
        
    def execute_pd(self, pipeline):
        if self.cache is not None:
            return
        _proc = self.get_function_pd()
        if not self.op.children or len(self.op.children) == 0:
            self.cache = _proc()
        else:
            child_output = pipeline[self.op.children[0]].cache
            self.cache = _proc(child_output)
            
    def execute_spark(self, pipeline, rdp):
        if self.cache is not None:
            return
        _proc = self.get_function_spark(rdp)
        if not self.op.children or len(self.op.children) == 0:
            self.cache = _proc()
        else:
            child_output = pipeline[self.op.children[0]].cache
            self.cache = _proc(child_output)