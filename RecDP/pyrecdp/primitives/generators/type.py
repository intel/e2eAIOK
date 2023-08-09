from .base import BaseFeatureGenerator as super_class
from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import pandas as pd
import numpy as np
from pandas.api import types as pdt
import copy
from featuretools.primitives.base import TransformPrimitive

class TypeCheckFeatureGenerator(super_class):
    def __init__(self, final = False, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        config = {}
        for idx in range(len(pa_schema)):
            pa_field = pa_schema[idx]
            if pa_field.is_categorical_and_string:
                pa_schema[idx] = SeriesSchema(pa_field.name, pd.StringDtype())
                config[pa_field.name] = 'str'
            elif pa_field.is_categorical:
                pa_schema[idx] = SeriesSchema(pa_field.name, pd.Int32Dtype())
                config[pa_field.name] = 'int'
        
        cur_idx = max_idx
        #pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'type_check', config = config)
        return pipeline, cur_idx, cur_idx

class IntTransformer(TransformPrimitive):
    name = "astype_int"
    return_type = int

    def get_function(self):
        def astype_int(array):
            try:
                ret = array.astype(int)
            except:
                ret = array
            return ret
        return astype_int

class FloatTransformer(TransformPrimitive):
    name = "astype_float"
    return_type = float

    def get_function(self):
        def astype_float(array):
            try:
                ret = array.astype(float)
            except:
                ret = array
            return ret
        return astype_float

class TypeConvertFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, final = False, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        feature_in = {}
        feature_in_out_map = {}
        is_useful = False
        for pa_field in pa_schema:
            if pa_field.is_integer:
                op_class = IntTransformer
                feature_in[pa_field.name] = op_class
            if pa_field.is_float:
                op_class = FloatTransformer
                feature_in[pa_field.name] = op_class

        for in_feat_name, op_clz in feature_in.items():
            is_useful = True
            feature_in_out_map[in_feat_name] = []
            feature_in_out_map[in_feat_name].append((in_feat_name, op_clz))
        if is_useful:
            cur_idx = max_idx + 1
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = "astype", config = feature_in_out_map)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx
