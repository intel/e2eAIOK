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

cur_dir = str(Path(__file__).parent.resolve())
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
    
    def test_TextReadWrite(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            JsonlWriter("tests/data/llm_data/tiny_c4_sample_out.jsonl") 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)

    def test_TextNormalize(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            TextNormalize() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextBytesize(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            TextBytesize() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextSourceId(self):
        pipeline = TextPipeline()
        ops = [
            SourcedJsonlReader("tests/data/llm_data/", source_prefix="")
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextSourceIdParquet(self):
        pipeline = TextPipeline()
        ops = [
            SourcedParquetReader("tests/data/llm_data/", source_prefix="")
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextLengthFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            LengthFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextBadwordsFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            BadwordsFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextProfanityFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            ProfanityFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextFixer(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            TextFix() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextLanguageIdentify(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")) 
        ]
        pipeline.add_operations(ops)

        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextDocumentSplit(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            DocumentSplit() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextPIIRemoval(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")) 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextURLFilter(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/tiny_c4_sample.jsonl"),
            #JsonlReader("tests/data/llm_data/PILE/NIH_sample.jsonl"),
            URLFilter() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextFuzzyDeduplicate(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/PILE/NIH_sample.jsonl"),
            FuzzyDeduplicate() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextGlobalDeduplicate(self):
        pipeline = TextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/PILE/NIH_sample.jsonl"),
            GlobalDeduplicate() 
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        pd = RDS(ret).to_pandas()
        display(pd)
        
    def test_TextPIIRemoval_resumable(self):
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface"))
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        
    def test_TextPipeline_resumable(self):
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