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

from pyrecdp.core.schema import SeriesSchema
from pyrecdp.primitives.operations import Operation
import pandas as pd
import numpy as np
from pandas.api import types as pdt
import copy
from featuretools.primitives.base import TransformPrimitive
from tqdm import tqdm

class TimeSeriesInferFeatureProfiler():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx, df, y = None, ts = None):
        time_series_features = []
        pa_schema = pipeline[children[0]].output
        for pa_field, feature_name in tqdm(zip(pa_schema, df.columns), total=len(df.columns)):
            if pa_field.is_timeseries:
                time_series_features.append(pa_field)
        
        for pa_field, feature_name in tqdm(zip(pa_schema, df.columns), total=len(df.columns)):
            config = {}
            if pa_field.is_categorical:
                for ts in time_series_features:
                    if pa_field.name == ts.name:
                        continue
                    group_id = f"{ts.name}_{pa_field.name}_idx"
                    config['group_id'] = [group_id]
                    config['is_grouped_categorical'] = True
                    config_tmp = {}
                    config_tmp['group_id'] = [group_id]
                    config_tmp['is_grouped_categorical'] = True
                    ts.copy_config_from(config_tmp)
                pa_field.copy_config_from(config)
        
        # append to pipeline
        cur_idx = max_idx + 1
        config = [x.mydump() for x in pa_schema]
        pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'time_series_infer', config = config)
        return pipeline, cur_idx, cur_idx
    
