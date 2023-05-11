from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
from pyrecdp.core import DataFrameSchema
from pyrecdp.core.utils import sample_read
from pyrecdp.primitives.operations import Operation
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class RelationalBuilder(BasePipeline):
    def __init__(self, dataset, label, *args, **kwargs):
        super().__init__(dataset, label)

        # If we provided multiple datasets in this workload
        self.generators.append([cls() for cls in relation_builder_list])
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
        child = self.children
        max_id = max(self.children)
        for i in range(len(self.generators)):
            for generator in self.generators[i]:
                child = child if isinstance(child, list) else [child]
                self.pipeline, child, max_id = generator.fit_prepare(self.pipeline, child, max_id)
        return child, max_id