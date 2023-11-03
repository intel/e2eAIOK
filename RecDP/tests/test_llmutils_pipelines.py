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
        
    def tearDown(self) -> None:
        print("Test completed, view results and delete output")
        import pandas as pd
        import os
        import shutil
        output_path = "ResumableTextPipeline_output"
        try:
            dir_name_list = [i for i in os.listdir("ResumableTextPipeline_output") if i.endswith('jsonl')]
            for dir_name in dir_name_list:
                print(dir_name)
                tmp_df = pd.read_parquet(os.path.join("ResumableTextPipeline_output", dir_name))
                print(f"total num_samples is {len(tmp_df)}")
                display(tmp_df.head())
            shutil.rmtree("ResumableTextPipeline_output")
        except Exception as e:
            print(e)
        return super().tearDown()

    def test_ResumableTextPipeline(self):
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            LengthFilter(),
            ProfanityFilter(),
            LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_import(self):
        pipeline = ResumableTextPipeline(pipeline_file = 'tests/data/import_test_pipeline.json')
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_customerfilter_op(self):
        def cond(text):
            return text > 0.9
        
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            TextQualityScorer(),
            TextCustomerFilter(cond, text_key='doc_score'),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.plot()
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_customermap_op(self):
        def classify(text):
            return 1 if text > 0.8 else 0
        
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            TextQualityScorer(),
            TextCustomerMap(classify, text_key='doc_score'),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.plot()
        pipeline.execute()
        del pipeline
        
    def test_ResumableTextPipeline_customer_function(self):
        def proc(text):
            return f'processed_{text}'
        
        pipeline = ResumableTextPipeline()
        pipeline.add_operation(JsonlReader("tests/data/llm_data/"))
        pipeline.add_operation(proc, text_key='text')
        pipeline.add_operation(PerfileParquetWriter("ResumableTextPipeline_output"))
        pipeline.plot()
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_with_fuzzyDedup(self):
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            TextQualityScorer(),
            FuzzyDeduplicate(),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_with_globalDedup(self):
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            ProfanityFilter(),
            GlobalDeduplicate(),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_with_bothDedup(self):
        pipeline = ResumableTextPipeline()
        #pipeline.enable_statistics()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            TextQualityScorer(),
            FuzzyDeduplicate(),
            GlobalDeduplicate(),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline
        
    def test_ResumableTextPipeline_with_bothDedup_withLog(self):
        pipeline = ResumableTextPipeline()
        pipeline.enable_statistics()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            TextQualityScorer(),
            FuzzyDeduplicate(),
            GlobalDeduplicate(),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_load_pipeline_from_yaml_file(self):
        pipeline = ResumableTextPipeline("tests/config/llm/pipeline/pipeline_example.yaml")
        pipeline.execute()
        del pipeline
        
    def test_pipeline_execute_pandasdf_ray(self):
        import pandas as pd
        pipeline = TextPipeline()
        ops = [
            LengthFilter(),
            ProfanityFilter(),
            LanguageIdentify(fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin"))
        ]
        pipeline.add_operations(ops)
        df = pd.read_parquet("tests/data/PILE/NIH_sample.parquet")
        ret = pipeline.execute(df)
        display(ret.to_pandas())
        
    def test_pipeline_execute_pandasdf_spark(self):
        import pandas as pd
        pipeline = TextPipeline()
        ops = [
            FuzzyDeduplicate(),
        ]
        pipeline.add_operations(ops)
        df = pd.read_parquet("tests/data/PILE/NIH_sample.parquet")
        ret = pipeline.execute(df)
        display(ret.toPandas())

