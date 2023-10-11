import unittest
import sys
import pandas as pd
from pathlib import Path
import os
from IPython.display import display

pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)
from pyrecdp.primitives.operations import *
from pyrecdp.LLM import TextPipeline, ResumableTextPipeline
import json
from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
from pyspark.sql import DataFrame
import psutil
import ray
from pyrecdp.core import SparkDataProcessor

class RDS:
    def __init__(self, ds):
        self.ds_engine = 'spark' if isinstance(ds, DataFrame) else 'ray'
        self.ds = ds
    def to_pandas(self):
        print(self.ds_engine)
        if self.ds_engine == 'ray':
            return self.ds.to_pandas()
        elif self.ds_engine == 'spark':
            return self.ds.toPandas()
        else:
            pass
     
class Test_LLMUtils_Pipeline(unittest.TestCase):
    
    def setUp(self) -> None:
        print(f"\n******\nTesting Method Name: {self._testMethodName}\n******")
        
    def test_TextPIIRemoval_resumable(self):
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface"))
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_TextPipeline_with_mode(self):
        pipeline = TextPipeline.init(mode='resumable')
        ops = [
            JsonlReader("tests/data/llm_data/"),
            LengthFilter(),
            ProfanityFilter(),
            LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")),
            PerfileParquetWriter("ResumableTextPipeline_output_20231004205724")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_TextPipeline_import_with_mode(self):
        pipeline = TextPipeline.init(mode='resumable', pipeline_file = 'tests/data/import_test_pipeline.json')
        pipeline.execute()
        del pipeline
