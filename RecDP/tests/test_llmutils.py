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
        
    def test_quality_classifier(self):
        from pyrecdp.primitives.llmutils import quality_classifier
        file_path = os.path.join(cur_dir, "data/llm_data/arxiv_sample_100.jsonl")
        save_path = os.path.join(cur_dir, "data/output/qualify_classify")
        quality_classifier(file_path, save_path, "jsonl")

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
            output_dataset = pii_remove(dataset=spark_df,text_column="text", model_root_path=model_root_path, show_secret_column=True)
            ctx.show(output_dataset)

    def test_language_identify_spark(self):
        from pyrecdp.primitives.llmutils import language_identify_spark
        from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
        with SparkContext("tests/data/llm_data/tiny_c4_sample.jsonl") as ctx:
            spark_df = ctx.ds
            fasttext_model_dir = os.path.join(RECDP_MODELS_CACHE, "lid.bin")
            lid_df = language_identify_spark(spark_df, fasttext_model_dir)
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
    #from pyrecdp.primitives.llmutils import shrink_document_MP, text_to_jsonl_MP, global_hash_mp, global_dedup
    def test_llm_ray_near_dedup(self):
        from pyrecdp.core.utils import Timer
        import shutil, argparse, pickle
        from pyrecdp.primitives.llmutils.utils import read_json
        from pyrecdp.primitives.llmutils.near_dedup import minHashLSH_prepare, generate_connected_components, generate_duplicates_dict
        from pyrecdp.primitives.llmutils.shrink_jsonl import shrink_document_MP
        
        data_dir = "tests/data/llm_data"
        data_files = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        dup_dir = "tests/data/PILE_dup_out"
        out_dir = "tests/data/PILE_dedup"
        ngram_size = 13
        num_perm = 256
        bands = 9
        ranges = 13
        with SparkContext() as ctx:
            spark = ctx.spark
            with Timer("Load data with RowID"):
                df = read_json(data_files, spark).cache()
                total_length = df.count()

            pipeline = minHashLSH_prepare(df, num_perm, ngram_size, bands, ranges)
            with Timer("generate minHashLsh"):
                if os.path.exists(dup_dir):
                    shutil.rmtree(dup_dir, ignore_errors=True)
                results = pipeline.saveAsTextFile(dup_dir)
                
            
            with Timer(f"generate_connected_components all"):
                dup_connected_args = argparse.Namespace()
                dup_connected_args.input_dir = dup_dir
                dup_connected_args.out_file = os.path.join(
                    dup_dir, "connected_components.pickle"
                )
                generate_connected_components.generate_connected_components_mp(
                    dup_connected_args
                )
                
            with Timer(f"generate_duplicates_dict all"):
                dup_docs = os.path.join(dup_dir, "duplicates.pickle")
                dup_dict_args = argparse.Namespace()
                dup_dict_args.input_file = os.path.join(
                    dup_dir, "connected_components.pickle"
                )
                dup_dict_args.out_file = dup_docs
                generate_duplicates_dict.generate_duplicates(dup_dict_args)
                
            dup_dict = pickle.load(open(os.path.join(dup_dir, "duplicates.pickle"), 'rb'))
            dup_sum = 0
            for _, v in dup_dict.items():
                dup_sum += len(list(v))

            dup_dict = os.path.join(dup_dir, "duplicates.pickle")
            out_dir = os.path.join(dup_dir, "output")
            with Timer("remove duplicate documents"):
                shrink_document_MP(data_dir, dup_dict, out_dir)

            print(f"Completed!!")
            print(f"    total processed {total_length} documents")
            print(f"    total detected {dup_sum} duplicated documents")
            print(f"    duplicate ratio is {dup_sum/total_length}")


    def test_llm_ray_convert_jsonl(self):
        from pyrecdp.primitives.llmutils.text_to_jsonl import text_to_jsonl_MP
        data_dir = "tests/data/pmc"
        out_dir = "tests/data/pmc_jsonl"
        text_to_jsonl_MP(data_dir, out_dir, 2)


    def test_llm_ray_global_hash_jsonl(self):
        import pandas as pd
        from pyrecdp.primitives.llmutils.global_hash import global_hash_mp
        from pyrecdp.core.utils import Timer
        source = 'PILE'
        in_type = 'jsonl'
        n_parallel = 4
        out_dir = "tests/data/global_hash_out/"
        is_norm = True
        data_dir = "tests/data/PILE"
        with Timer("execute global_hash_mp"):
            global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm)
        
        from pyrecdp.primitives.llmutils.global_dedup import get_hash_indexing
        out_dir = "tests/data/global_hash_index"
        data_dir = "tests/data/global_hash_out/"
        with Timer("execute get_hash_indexing"):
            get_hash_indexing(data_dir, out_dir)
        
        from pyrecdp.primitives.llmutils.global_dedup import combine_hash_indexing
        out_dir = "tests/data/combined_hash_index/"
        data_dir_dir = ["tests/data/global_hash_index/"]
        with Timer("execute combine_hash_indexing"):
            combine_hash_indexing(data_dir_dir, out_dir)
        
        from pyrecdp.primitives.llmutils.global_dedup import get_duplication_list
        data_dir = "tests/data/global_hash_index"
        out_dir = "tests/data/duplications_index"
        with Timer("execute get_duplication_list"):
            get_duplication_list(data_dir, out_dir)
        
        from pyrecdp.primitives.llmutils import index_based_reduction
        in_dir = "tests/data/global_hash_out"
        dup_dir = "tests/data/duplications_index"
        out_dir = "tests/data/global_dedup/deduplicated"
        with Timer("execute index_based_reduction"):
            index_based_reduction(in_dir, dup_dir, out_dir)
        
        pdf = pd.read_parquet(out_dir)
        display(pdf)
        
    
    def test_llm_ray_global_hash_parquet(self):
        import pandas as pd
        from pyrecdp.primitives.llmutils.global_hash import global_hash_mp
        from pyrecdp.core.utils import Timer
        source = 'PILE'
        in_type = 'parquet'
        n_parallel = 4
        out_dir = "tests/data/global_hash_out/"
        is_norm = True
        data_dir = "tests/data/PILE"
        with Timer("execute global_hash_mp"):
            global_hash_mp(source, data_dir, in_type, n_parallel, out_dir, is_norm)
        
        from pyrecdp.primitives.llmutils.global_dedup import get_hash_indexing
        out_dir = "tests/data/global_hash_index"
        data_dir = "tests/data/global_hash_out/"
        with Timer("execute get_hash_indexing"):
            get_hash_indexing(data_dir, out_dir)
        
        from pyrecdp.primitives.llmutils.global_dedup import combine_hash_indexing
        out_dir = "tests/data/combined_hash_index/"
        data_dir_dir = ["tests/data/global_hash_index/"]
        with Timer("execute combine_hash_indexing"):
            combine_hash_indexing(data_dir_dir, out_dir)
        
        from pyrecdp.primitives.llmutils.global_dedup import get_duplication_list
        data_dir = "tests/data/global_hash_index"
        out_dir = "tests/data/duplications_index"
        with Timer("execute get_duplication_list"):
            get_duplication_list(data_dir, out_dir)
        
        from pyrecdp.primitives.llmutils import index_based_reduction
        in_dir = "tests/data/global_hash_out"
        dup_dir = "tests/data/duplications_index"
        out_dir = "tests/data/global_dedup/deduplicated"
        with Timer("execute index_based_reduction"):
            index_based_reduction(in_dir, dup_dir, out_dir)
        
        pdf = pd.read_parquet(out_dir)
        display(pdf)


    def test_llm_ray_global_dedup(self):
        from pyrecdp.primitives.llmutils.global_dedup import global_dedup
        import pandas as pd
        in_type = 'jsonl'
        out_dir = "tests/data/global_dedup/"
        data_dir = "tests/data/PILE"
        global_dedup(data_dir, out_dir, "PILE", in_type)
        pdf = pd.read_parquet(out_dir + 'deduplicated')
        display(pdf)

    



    