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
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            TextNormalize() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextLengthFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            LengthFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextBadwordsFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            BadwordsFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextProfanityFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            ProfanityFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextFixer(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            TextFix() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextLanguageIdentify(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")) 
        ]
        pipeline.add_operations(ops)

        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextDocumentSplit(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            DocumentSplit() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)
        
    def test_TextPIIRemoval(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")) 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = ret.to_pandas()
        display(pd)