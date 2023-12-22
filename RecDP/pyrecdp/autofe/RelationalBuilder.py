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

from pyrecdp.primitives.generators import *
from pyrecdp.primitives.profilers import *
from pyrecdp.autofe.TabularPipeline import TabularPipeline
from pyrecdp.core.schema import DataFrameSchema
from pyrecdp.core.utils import sample_read
from pyrecdp.primitives.operations import Operation
from pyrecdp.core.dataframe import DataFrameAPI
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class RelationalBuilder(TabularPipeline):
    def __init__(self, dataset, label, *args, **kwargs):
        super().__init__(dataset, label)

        # If we provided multiple datasets in this workload
        self.generators.append([cls() for cls in relation_builder_list])
        self.data_profilers = [cls() for cls in feature_infer_list]
        idx = 1
        self.children = [0]
        for table_name, table in self.supplementary.items():
            if isinstance(table, str):
                data = sample_read(table)
                self.pipeline[idx] = Operation(
                    idx, None, output = DataFrameSchema(data), op = 'DataLoader', config = {'table_name': table_name, 'file_path': table})
            else:
                self.pipeline[idx] = Operation(
                    idx, None, output = DataFrameSchema(table), op = 'DataFrame', config = table_name)
            self.children.append(idx)
            idx += 1
        self.fit_analyze()

    def fit_analyze(self, *args, **kwargs):
        children = []
        max_id = max(self.children)
        for child in self.children:
            op = self.pipeline[child]
            if op.op == 'DataLoader':
                table = op.config['table_name']
            else:
                table = op.config
            X = DataFrameAPI().instiate(self.dataset[table])
            sampled_data = X.may_sample()
            for profiler in self.data_profilers:
                self.pipeline, child, max_id = profiler.fit_prepare(self.pipeline, [child], max_id, sampled_data, self.y)
            children.append(max_id)
        for i in range(len(self.generators)):
            for generator in self.generators[i]:
                children = children if isinstance(children, list) else [children]
                self.pipeline, children, max_id = generator.fit_prepare(self.pipeline, children, max_id)
        return children, max_id