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
import json
from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE

cur_dir = str(Path(__file__).parent.resolve())

def convert_json(s):
    if isinstance(s, str):
        content = json.loads(s)
    elif isinstance(s, dict):
        if len(s.keys()) == 1:
            content = s[next(iter(s))]
            if isinstance(content, str):
                content = json.loads(content)
        else:
            content = s
    content['text'] = str(content['text'])
    if 'meta' in content:
        content['meta'] = str(content['meta'])       
    return content

class Test_LLMUtils_Pipeline(unittest.TestCase):
    
    def setUp(self) -> None:
        pass

    def test_TextNormalize(self):
        pipeline = TextPipeline()
        ops = [
            TextNormalize() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_json("tests/data/llm_data/PILE/NIH_sample.jsonl")
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextLengthFilter(self):
        pipeline = TextPipeline()
        ops = [
            LengthFilter() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_json("tests/data/llm_data/arxiv_sample_100.jsonl")
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextBadwordsFilter(self):
        pipeline = TextPipeline()
        ops = [
            BadwordsFilter() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_text("tests/data/llm_data/tiny_c4_sample.jsonl")
        dataset = dataset.map(convert_json)
        display(dataset)
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextProfanityFilter(self):
        pipeline = TextPipeline()
        ops = [
            ProfanityFilter() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_text("tests/data/llm_data/tiny_c4_sample.jsonl")
        dataset = dataset.map(convert_json)
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextFixer(self):
        pipeline = TextPipeline()
        ops = [
            TextFix() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_text("tests/data/llm_data/tiny_c4_sample.jsonl")
        dataset = dataset.map(convert_json)
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextLanguageIdentify(self):
        pipeline = TextPipeline()
        ops = [
            LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")) 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_text("tests/data/llm_data/tiny_c4_sample.jsonl")
        dataset = dataset.map(convert_json)
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextDocumentSplit(self):
        pipeline = TextPipeline()
        ops = [
            DocumentSplit() 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_text("tests/data/llm_data/tiny_c4_sample.jsonl")
        dataset = dataset.map(convert_json)
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextPIIRemoval(self):
        pipeline = TextPipeline()
        ops = [
            PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")) 
        ]
        pipeline.add_operations(ops)
        
        dataset = rd.read_text("tests/data/llm_data/tiny_c4_sample.jsonl")
        dataset = dataset.map(convert_json)
        ret = pipeline.execute(dataset)
        pd = ret.to_pandas()
        display(pd)