import unittest
import sys
import pandas as pd
from pathlib import Path
import os

pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)


from pyrecdp.primitives.llmutils import near_dedup, near_dedup_spk, shrink_document_MP, text_to_jsonl_MP, pii_remove, \
    filter_by_blocklist, language_identify, language_identify_spark, Classifier, profanity_filter, filter_by_bad_words, filter_by_length, global_hash

from pyrecdp.primitives.llmutils.utils import get_target_file_list


cur_dir = str(Path(__file__).parent.resolve())


class Test_LLMUtils(unittest.TestCase):
    def setUp(self):
        self.data_files = ["tests/data/llm_data/NIH_sample.jsonl"]
        self.data_dir = "tests/data/llm_data/"
        self.dup_dir = "./near_dedup/"
        self.fasttext_model = "/home/vmagent/models/lid.bin"  # Only used for github CICD test.

    def test_near_dedup(self):
        data_files = self.data_files
        dup_dir = self.dup_dir
        ngram_size = 13
        num_perm = 256
        bands = 9
        ranges = 13
        near_dedup(data_files, dup_dir, ngram_size, num_perm, bands, ranges)

    def test_near_dedup_spark(self):
        from pyrecdp.core import SparkDataProcessor
        from pyspark.sql.types import StructType, StructField, StringType
        import pyspark.sql.functions as F
        data_files = self.data_files
        dup_dir = self.dup_dir
        ngram_size = 13
        num_perm = 256
        bands = 9
        ranges = 13
        rdp = SparkDataProcessor()
        spark = rdp.spark
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("meta", StringType(), True)
        ])
        spark_df = spark.read.text(data_files)
        spark_df = spark_df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*")
        print("input is ")
        spark_df.show()
        ret_df = near_dedup_spk(spark_df, ngram_size, num_perm, bands, ranges)
        print("output is")
        ret_df.show()
        
    def test_global_hash_jsonl(self):
        source = 'PILE'
        data_files = ["NIH_sample.jsonl"]
        in_type = 'jsonl'
        n_parallel = 4
        out_dir = "tests/data/llm_data/global_hash_out/"
        is_norm = True
        data_dir = self.data_dir
        global_hash(source, data_files, data_dir, in_type, n_parallel, out_dir, is_norm)
        
    def test_global_hash_parquet(self):
        source = 'PILE'
        data_files = ["NIH_sample.parquet"]
        in_type = 'parquet'
        n_parallel = 4
        out_dir = "tests/data/llm_data/global_hash_out/"
        is_norm = True
        data_dir = self.data_dir
        global_hash(source, data_files, data_dir, in_type, n_parallel, out_dir, is_norm)

    def test_shrink_jsonl(self):
        data_dir = self.data_dir
        dup_dir = self.dup_dir
        dup_dict = os.path.join(dup_dir, "duplicates.pickle")
        out_dir = os.path.join(dup_dir, "output")
        shrink_document_MP(data_dir, dup_dict, out_dir)

    def test_text_to_jsonl(self):
        data_dir = "tests/data/llm_data/pmc"
        out_dir = "pmc_jsonl"
        text_to_jsonl_MP(data_dir, out_dir, 2)

    def test_ppi_remove(self):
        from pyrecdp.core import SparkDataProcessor

        sparkDP = SparkDataProcessor()
        spark=sparkDP.spark
        input_dataset = spark.read.load(path="tests/data/llm_data/arxiv_sample_100.jsonl", format="json")
        output_dataset = pii_remove(input_dataset)
        output_dataset.write.save(path="./tmp", format="json", mode="overwrite")

    def test_filter_jsonl(self):
        data_dir = "tests/data/llm_data"
        out_dir = "tests/data/filter_out"
        filter_by_blocklist(data_dir, out_dir)

    def test_profanity_filter(self):
        data_dir = "tests/data/llm_data"
        out_dir = "tests/data/filter_out"
        profanity_filter(data_dir, out_dir)

    def test_bad_words_filter(self):
        data_dir = "tests/data/llm_data"
        out_dir = "tests/data/filter_out"
        filter_by_bad_words(data_dir, out_dir)

    def test_length_filter(self):
        data_dir = "tests/data/llm_data"
        out_dir = "tests/data/filter_out"
        filter_by_length(data_dir, out_dir)


    def test_language_identify(self):
        data_dir = os.path.join(cur_dir, "data/llm_data")
        data_files = get_target_file_list(data_dir, "jsonl", "file://")
        fasttext_model_dir = self.fasttext_model
        language_identify_output_dir = os.path.join(data_dir, "language_identify")
        language_identify(data_dir, data_files, fasttext_model_dir, 'text', 'lang', language_identify_output_dir, "file://")

    def test_language_identify_spark(self):
        from pyrecdp.core import SparkDataProcessor
        fasttext_model_dir = self.fasttext_model
        data_dir = os.path.join(cur_dir, "data/llm_data")
        data_file = self.data_files[0]
        save_path = os.path.join(data_dir, "language_identify/lid_df")
        rdp = SparkDataProcessor()
        spark=rdp.spark
        spark_df = spark.read.json(data_file)
        print("input is ")
        spark_df.show()
        lid_df = language_identify_spark(spark_df, fasttext_model_dir, 'text', 'lang', save_path)
        print("output is")
        lid_df.show()
