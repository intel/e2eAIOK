import unittest
import sys
import pandas as pd
from pathlib import Path
import os
from IPython.display import display

from pyrecdp.primitives.llmutils.text_fixer import text_fixer

pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)
from pyrecdp.primitives.llmutils import near_dedup, near_dedup_spk, shrink_document_MP, text_to_jsonl_MP, pii_remove, \
    filter_by_blocklist, language_identify, language_identify_spark, profanity_filter, filter_by_bad_words, \
    filter_by_length, global_hash_mp, global_dedup, global_dedup_spk, \
    classify, classify_spark
from pyrecdp.primitives.llmutils.utils import get_target_file_list

cur_dir = str(Path(__file__).parent.resolve())

class Test_LLMUtils(unittest.TestCase):
    def setUp(self):
        self.data_files = ["tests/data/llm_data/PILE/NIH_sample.jsonl"]
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
        in_type = 'jsonl'
        n_parallel = 4
        out_dir = "tests/data/llm_data/global_hash_out/"
        is_norm = True
        data_dir = "tests/data/llm_data/PILE"
        global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_global_hash_parquet(self):
        source = 'PILE'
        in_type = 'parquet'
        n_parallel = 4
        out_dir = "tests/data/llm_data/global_hash_out/"
        is_norm = True
        data_dir = "tests/data/llm_data/PILE"
        global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_get_hash_indexing(self):
        from pyrecdp.primitives.llmutils.global_dedup import get_hash_indexing
        out_dir = "tests/data/llm_data/global_hash_index"
        data_dir = "tests/data/llm_data/global_hash_out/"
        get_hash_indexing(data_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_combine_hash_indexing(self):
        from pyrecdp.primitives.llmutils.global_dedup import combine_hash_indexing
        out_dir = "tests/data/llm_data/combined_hash_index/"
        data_dir_dir = ["tests/data/llm_data/global_hash_index/"]
        combine_hash_indexing(data_dir_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_get_duplication_list(self):
        from pyrecdp.primitives.llmutils.global_dedup import get_duplication_list
        data_dir = "tests/data/llm_data/global_hash_index"
        out_dir = "tests/data/llm_data/duplications_index"
        get_duplication_list(data_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_index_based_reduction(self):
        from pyrecdp.primitives.llmutils import index_based_reduction
        in_dir = "tests/data/llm_data/global_hash_out"
        dup_dir = "tests/data/llm_data/duplications_index"
        out_dir = "tests/data/llm_data/global_dedup/deduplicated"
        index_based_reduction(in_dir, dup_dir, out_dir)
        pdf = pd.read_parquet(out_dir)
        display(pdf)

    def test_global_dedup(self):
        in_type = 'jsonl'
        out_dir = "tests/data/llm_data/global_dedup/"
        data_dir = "tests/data/llm_data/PILE"
        global_dedup(data_dir, out_dir, "PILE", in_type)
        pdf = pd.read_parquet(out_dir + 'deduplicated')
        display(pdf)

    def test_global_dedup_spark(self):
        from pyrecdp.core import SparkDataProcessor
        from pyspark.sql.types import StructType, StructField, StringType
        import pyspark.sql.functions as F
        data_files = self.data_files
        is_norm = True
        rdp = SparkDataProcessor()
        spark = rdp.spark
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("meta", StringType(), True)
        ])
        for data_file in data_files:
            spark_df = spark.read.text(data_file)
            spark_df = spark_df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*")
            source = os.path.basename(data_file).replace("/", "_")
            print("input is ")
            spark_df.show()
            ret = global_dedup_spk(spark_df, source, is_norm)
            ret.show()

    def test_shrink_jsonl(self):
        data_dir = "tests/data/llm_data/PILE"
        dup_dir = self.dup_dir
        dup_dict = os.path.join(dup_dir, "duplicates.pickle")
        out_dir = os.path.join(dup_dir, "output")
        shrink_document_MP(data_dir, dup_dict, out_dir)

    def test_text_to_jsonl(self):
        data_dir = "tests/data/llm_data/pmc"
        out_dir = "pmc_jsonl"
        text_to_jsonl_MP(data_dir, out_dir, 2)

    def test_pii_remove(self):
        from pyrecdp.core import SparkDataProcessor

        sparkDP = SparkDataProcessor()
        spark = sparkDP.spark
        input_dataset = spark.read.load(path="tests/data/llm_data/arxiv_sample_100.jsonl", format="json")
        output_dataset = pii_remove(input_dataset,text_column="text",keep_secret_column=True)
        output_dataset.write.save("tmp/pii", mode="overwrite", format="json")
        output_dataset.show(truncate=True)

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


    def test_text_fixer(self):
        data_dir = "tests/data/yao_data"
        out_dir = "tests/data/filter_out"
        in_type = "jsonl"
        text_types = ["html", 'latex', "codes"]
        text_fixer(data_dir, in_type, out_dir,text_types)
        
        
    def test_language_identify(self):
        data_dir = os.path.join(cur_dir, "data/llm_data/PILE")
        fasttext_model_dir = self.fasttext_model
        language_identify_output_dir = os.path.join(data_dir, "language_identify")
        language_identify(data_dir, "jsonl", fasttext_model_dir, 'text', 'lang', language_identify_output_dir, "file://")


    def test_language_identify_spark(self):
        from pyrecdp.core import SparkDataProcessor
        fasttext_model_dir = self.fasttext_model
        data_dir = os.path.join(cur_dir, "data/llm_data")
        data_file = self.data_files[0]
        save_path = os.path.join(data_dir, "language_identify/lid_df")
        rdp = SparkDataProcessor()
        spark = rdp.spark
        spark_df = spark.read.json(data_file)
        print("input is ")
        spark_df.show()
        lid_df = language_identify_spark(spark_df, fasttext_model_dir, 'text', 'lang', save_path, "file://")
        print("output is")
        lid_df.show()

    def test_classify(self):
        data_dir = os.path.join(cur_dir, "data/llm_data/PILE")
        fasttext_model_dir = self.fasttext_model
        lid_save_path = os.path.join(cur_dir, "data/output/lid")
        classify_save_path = os.path.join(cur_dir, "data/output/classify")
        language_identify(data_dir, "jsonl", fasttext_model_dir, 'text', 'lang', lid_save_path, "file://")
        classify(lid_save_path, "parquet", classify_save_path, "lang", "file://")

    def test_classify_spark(self):
        from pyrecdp.core import SparkDataProcessor
        fasttext_model_dir = self.fasttext_model
        data_file = os.path.join(cur_dir, "data/llm_data/arxiv_sample_100.jsonl")
        classify_save_path = os.path.join(cur_dir, "data/output/classify_spark")
        lid_save_path = os.path.join(cur_dir, "data/output/lid_spark")
        rdp = SparkDataProcessor()
        spark=rdp.spark
        spark_df = spark.read.json(data_file)
        lid_df = language_identify_spark(spark_df, fasttext_model_dir, 'text', 'lang', lid_save_path, "file://")

        print("input is ")
        spark_df.show()
        classify_spark(lid_df, "lang", classify_save_path, "file://")

    def test_convert_from_text(self):
        from pyrecdp.primitives.llmutils import convert
        data_dir = "tests/data/llm_data/pmc"
        out_dir = "tests/data/llm_data/pmc_parquet"
        convert(data_dir, 'text', 1, out_dir)
        pdf = pd.read_parquet(out_dir + "/part_0.parquet")
        display(pdf)

    def test_convert_from_jsonl(self):
        from pyrecdp.primitives.llmutils import convert
        data_dir = "tests/data/llm_data/PILE/"
        out_dir = "tests/data/llm_data/PILE_parquet_converted"
        convert(data_dir, 'jsonl', 1, out_dir)
        list_file = os.listdir(out_dir)
        print(f"out dir contains {list_file}")
        pdf = pd.read_parquet(out_dir)
        display(pdf)
        
    def test_text_normalization(self):
        from pyrecdp.primitives.llmutils import text_normalization
        data_dir = "tests/data/llm_data/PILE/"
        out_dir = "tests/data/llm_data/PILE_text_norm"
        text_normalization(data_dir, 'jsonl', out_dir)
        list_file = os.listdir(out_dir)
        print(f"out dir contains {list_file}")
        pdf = pd.read_parquet(out_dir)
        display(pdf)
        
    def test_text_normalization_spark(self):
        from pyrecdp.primitives.llmutils import text_normalization_spk
        from pyrecdp.core import SparkDataProcessor
        from pyspark.sql.types import StructType,StructField, StringType
        import pyspark.sql.functions as F
        data_file = "tests/data/llm_data/PILE/NIH_sample.jsonl"
        rdp = SparkDataProcessor()
        spark=rdp.spark
        schema = StructType([ 
            StructField("text",StringType(),True), 
            StructField("meta",StringType(),True)
        ])
        spark_df = spark.read.text(data_file)
        spark_df = spark_df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*")
        
        ret = text_normalization_spk(spark_df)
        ret.show()

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