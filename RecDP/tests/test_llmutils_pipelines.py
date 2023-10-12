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
from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
from pyrecdp.core import SparkDataProcessor

     
class Test_LLMUtils_Pipeline(unittest.TestCase):
    
    def setUp(self) -> None:
        print(f"\n******\nTesting Method Name: {self._testMethodName}\n******")

    def test_ResumableTextPipeline(self):
        pipeline = ResumableTextPipeline()
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

    def test_ResumableTextPipeline_import(self):
        pipeline = ResumableTextPipeline(pipeline_file = 'tests/data/import_test_pipeline.json')
        pipeline.execute()
        del pipeline
