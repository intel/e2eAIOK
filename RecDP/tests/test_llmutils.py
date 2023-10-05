import unittest
import os, sys
from pathlib import Path
from IPython.display import display

pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)

from pyrecdp.core import SparkDataProcessor
from pyrecdp.primitives.operations import JsonlReader

cur_dir = str(Path(__file__).parent.resolve())

class SparkContext:
    def __init__(self, dataset_path = None):
        self.dataset_path = dataset_path
        self.rdp = SparkDataProcessor()

    def __enter__(self):  
        self.spark = self.rdp.spark
        if self.dataset_path is not None:
            reader = JsonlReader(self.dataset_path)
            self.ds = reader.process_spark(self.spark)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
            
    def show(self, ds):
        pd = ds.toPandas()
        display(pd)
        
class Test_LLMUtils(unittest.TestCase):
    def setUp(self):
        print(f"\n******\nTesting Method Name: {self._testMethodName}\n******")
        self.data_files = ["tests/data/PILE/NIH_sample.jsonl"]
        self.dup_dir = "./near_dedup/"
        self.fasttext_model = "/home/vmagent/models/lid.bin"  # Only used for github CICD test.
        
    def test_quality_classifier(self):
        from pyrecdp.primitives.llmutils import quality_classifier
        file_path = os.path.join(cur_dir, "data/llm_data/arxiv_sample_100.jsonl")
        save_path = os.path.join(cur_dir, "data/output/qualify_classify")
        quality_classifier(file_path, save_path, overall_stats=True, file_system_prefix="file://")

    def test_quality_classifier_spark(self):
        from pyrecdp.primitives.llmutils import quality_classifier_spark
        from pyrecdp.core import SparkDataProcessor
        data_file = f'file://{os.path.join(cur_dir, "data/llm_data/arxiv_sample_100.jsonl")}'
        rdp = SparkDataProcessor()
        spark = rdp.spark
        spark_df = spark.read.json(data_file)
        quality_classifier_df = quality_classifier_spark(spark_df)
        quality_classifier_df.show()
        
    def test_diversity_analysis(self):
        from pyrecdp.primitives.llmutils import diversity
        data_dir = "tests/data/llm_data/"
        output_path = "tests/data/diversity_out"
        in_type = "jsonl"
        diversity(data_dir, in_type, output_path)
        
# ***** This test is to provide an example for EasyData ***** #
    def test_near_dedup_spark(self):
        from pyrecdp.primitives.llmutils import near_dedup_spk        
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            spark_df = ctx.ds
            ret_df = near_dedup_spk(spark_df)
            ctx.show(ret_df)
            
    def test_global_dedup_spark(self):
        from pyrecdp.primitives.llmutils import global_dedup_spk        
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            spark_df = ctx.ds
            ret_df = global_dedup_spk(spark_df)
            ctx.show(ret_df)
            
    def test_pii_remove_spark(self):
        from pyrecdp.primitives.llmutils import pii_remove
        from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
        with SparkContext("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl") as ctx:
            spark_df = ctx.ds
            model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")
            output_dataset = pii_remove(dataset=spark_df,text_column="text", model_root_path=model_root_path, show_secret_column=True, secret_column="secret")
            df = output_dataset.select("secret","__SECRETS__")
            ctx.show(df)
            for _, row in df.toPandas().iterrows():
                self.assertEqual(row["secret"], row["__SECRETS__"])

    def test_language_identify_spark(self):
        from pyrecdp.primitives.llmutils import language_identify_spark
        from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            spark_df = ctx.ds
            fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")
            lid_df = language_identify_spark(spark_df, fasttext_model_dir, 'text', 'lang')
            ctx.show(lid_df)

    def test_sentence_split_spark(self):
        from pyrecdp.primitives.llmutils import sentence_split
        import pandas as pd
        with SparkContext() as ctx:
            samples = [(
                    'Smithfield employs 3,700 people at its plant in Sioux Falls, '
                    'South Dakota. The plant slaughters 19,500 pigs a day — 5 '
                    'percent of U.S. pork.',
                    'Smithfield employs 3,700 people at its plant in Sioux Falls, '
                    'South Dakota.\nThe plant slaughters 19,500 pigs a day — 5 '
                    'percent of U.S. pork.')]
            spark_df = ctx.spark.createDataFrame(pd.DataFrame(samples, columns=["text", "target"]))
            ret_df = sentence_split(spark_df)
            ctx.show(ret_df)
            for _, row in ret_df.toPandas().iterrows():
                self.assertEqual(row["text"], row["target"])
        
################################################################

# This test is used to make sure our codes in llm-ray is still working
    from pyrecdp.primitives.llmutils import shrink_document_MP, text_to_jsonl_MP, global_hash_mp, global_dedup

    def test_near_dedup(self):
        data_files = self.data_files
        dup_dir = self.dup_dir
        ngram_size = 13
        num_perm = 256
        bands = 9
        ranges = 13
        near_dedup(data_files, dup_dir, ngram_size, num_perm, bands, ranges)

    
    def test_global_hash_jsonl(self):
        source = 'PILE'
        in_type = 'jsonl'
        n_parallel = 4
        out_dir = "tests/data/global_hash_out/"
        is_norm = True
        data_dir = "tests/data/PILE"
        global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_global_hash_parquet(self):
        source = 'PILE'
        in_type = 'parquet'
        n_parallel = 4
        out_dir = "tests/data/global_hash_out/"
        is_norm = True
        data_dir = "tests/data/PILE"
        global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_get_hash_indexing(self):
        from pyrecdp.primitives.llmutils.global_dedup import get_hash_indexing
        out_dir = "tests/data/global_hash_index"
        data_dir = "tests/data/global_hash_out/"
        get_hash_indexing(data_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_combine_hash_indexing(self):
        from pyrecdp.primitives.llmutils.global_dedup import combine_hash_indexing
        out_dir = "tests/data/combined_hash_index/"
        data_dir_dir = ["tests/data/global_hash_index/"]
        combine_hash_indexing(data_dir_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_get_duplication_list(self):
        from pyrecdp.primitives.llmutils.global_dedup import get_duplication_list
        data_dir = "tests/data/global_hash_index"
        out_dir = "tests/data/duplications_index"
        get_duplication_list(data_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_index_based_reduction(self):
        from pyrecdp.primitives.llmutils import index_based_reduction
        in_dir = "tests/data/global_hash_out"
        dup_dir = "tests/data/duplications_index"
        out_dir = "tests/data/global_dedup/deduplicated"
        index_based_reduction(in_dir, dup_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_global_dedup(self):
        in_type = 'jsonl'
        out_dir = "tests/data/global_dedup/"
        data_dir = "tests/data/PILE"
        global_dedup(data_dir, out_dir, "PILE", in_type)
        pdf = pd.read_parquet(out_dir + 'deduplicated')
        display(pdf)

    

    def test_shrink_jsonl(self):
        data_dir = "tests/data/PILE"
        dup_dir = self.dup_dir
        dup_dict = os.path.join(dup_dir, "duplicates.pickle")
        out_dir = os.path.join(dup_dir, "output")
        shrink_document_MP(data_dir, dup_dict, out_dir)

    def test_text_to_jsonl(self):
        data_dir = "tests/data/pmc"
        out_dir = "pmc_jsonl"
        text_to_jsonl_MP(data_dir, out_dir, 2)

    