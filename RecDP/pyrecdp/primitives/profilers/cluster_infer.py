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
from scipy.stats import ks_2samp

def try_even_distribution(s):
    df = s.to_frame()
    df['index'] = df.index
    distribution = df.groupby(by = s.name)['index'].agg(['count'])['count'].to_numpy()
    even_distribution = np.random.uniform(np.mean(distribution) * 0.9, np.mean(distribution) * 1.1, len(distribution))
    print(distribution)
    #print(even_distribution)
    print(f"{s.name} even_distribution is {ks_2samp(distribution,even_distribution)}")
    return False

class DistributionInferFeatureProfiler():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def fit_prepare(self, pipeline, children, max_idx, df, y = None, ts = None):
        ret_pa_fields = []
        pa_schema = pipeline[children[0]].output
        for pa_field, feature_name in tqdm(zip(pa_schema, df.columns), total=len(df.columns)):
            config = {}
            # Add y_label to y column
            if feature_name == y:
                continue
            if try_even_distribution(df[feature_name]):
                config['is_context'] = True
            
            pa_field.copy_config_from(config)
            ret_pa_fields.append(pa_field)
        
        # append to pipeline
        cur_idx = max_idx + 1
        config = [x.mydump() for x in ret_pa_fields]
        pipeline[cur_idx] = Operation(cur_idx, children, ret_pa_fields, op = 'distribution_infer', config = config)
        return pipeline, cur_idx, cur_idx
    
