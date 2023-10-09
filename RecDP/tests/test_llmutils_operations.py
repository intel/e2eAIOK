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

total_mem = int(psutil.virtual_memory().total * 0.5)
total_cores = psutil.cpu_count(logical=False)

class RayContext:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __enter__(self):
        if not ray.is_initialized():
            try:
                ray.init(object_store_memory=total_mem, num_cpus=total_cores)
            except:
                ray.init()
                
        reader = JsonlReader(self.dataset_path)
        self.ds = reader.process_rayds(None)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if ray.is_initialized():
            ray.shutdown()
            
    def show(self, ds):
        pd = ds.to_pandas()
        display(pd)
        
class SparkContext:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.rdp = SparkDataProcessor()

    def __enter__(self):  
        self.spark = self.rdp.spark             
        reader = JsonlReader(self.dataset_path)
        self.ds = reader.process_spark(self.spark)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
            
    def show(self, ds):
        pd = ds.toPandas()
        display(pd)

    
class Test_LLMUtils_Operations(unittest.TestCase):
    def setUp(self):
        print(f"\n******\nTesting Method Name: {self._testMethodName}\n******")

    ### ======  Ray ====== ###
    
    def test_bytesize_ray(self):
        op = TextBytesize() 
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_fuzzy_deduplication_ray(self):
        pass
        # Ray version not supported yet
        # op = FuzzyDeduplicate()
        # with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
        #     ctx.show(op.process_rayds(ctx.ds))
        
    def test_global_deduplication_ray(self):
        pass
        # Ray version not supported yet
        # op = GlobalDeduplicate()
        # with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
        #     ctx.show(op.process_rayds(ctx.ds))
            
    def test_filter_by_length_ray(self):
        op = LengthFilter()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_filter_by_badwords_ray(self):
        op = BadwordsFilter()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_filter_by_profanity_ray(self):
        op = ProfanityFilter()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_filter_by_url_ray(self):
        pass
        # Ray version not supported yet
        op = URLFilter()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_text_fixer_ray(self):
        op = TextFix()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_language_identify_ray(self):
        op = LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin"))
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))
     
    def test_text_normalization_ray(self):
        op = TextNormalize()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    def test_pii_removal_ray(self):
        op = PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface"))
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))
            
    def test_sentence_split_ray(self):
        op = DocumentSplit()
        with RayContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_rayds(ctx.ds))

    ### ======  Spark ====== ###
            
    def test_bytesize_spark(self):
        op = TextBytesize() 
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_fuzzy_deduplication_spark(self):
        op = FuzzyDeduplicate()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))
        
    def test_global_deduplication_spark(self):
        op = GlobalDeduplicate()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))
            
    def test_filter_by_length_spark(self):
        op = LengthFilter()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_filter_by_badwords_spark(self):
        op = BadwordsFilter()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_filter_by_profanity_spark(self):
        op = ProfanityFilter()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_filter_by_url_spark(self):
        op = URLFilter()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_text_fixer_spark(self):
        op = TextFix()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_language_identify_spark(self):
        op = LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin"))
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))
     
    def test_text_normalization_spark(self):
        op = TextNormalize()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_pii_removal_spark(self):
        # TODO: chendi found an performance issue when handling larger files in spark version
        op = PIIRemoval(model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface"))
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
        #with SparkContext("tests/data/PILE/NIH_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))
            
    def test_sentence_split_spark(self):
        op = DocumentSplit()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

    def test_qualityscore_spark(self):
        op = TextQualityScorer()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))
            
    def test_diversityindicate_spark(self):
        op = TextDiversityIndicate()
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            ctx.show(op.process_spark(ctx.spark, ctx.ds))

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

    def test_TextSourceId(self):
        pipeline = TextPipeline()
        ops = [
            SourcedJsonlReader("tests/data/llm_data/", source_prefix="")
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