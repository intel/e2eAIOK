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
from pyrecdp.LLM import TextPipeline
import ray.data as rd

cur_dir = str(Path(__file__).parent.resolve())

class Test_LLMUtils_Pipeline(unittest.TestCase):

    def test_TextPipeline(self):
        pipeline = TextPipeline()
        ops = [
            TextNormalize() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_json("tests/data/llm_data/PILE/NIH_sample.jsonl")
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)