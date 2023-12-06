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
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.rdp = SparkDataProcessor()

    def __enter__(self):
        self.spark = self.rdp.spark
        if self.dataset_path is not None:
            reader = JsonlReader(self.dataset_path)
            self.ds = reader.process_spark(self.spark)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del self.rdp

    def show(self, ds):
        pd = ds.toPandas()
        display(pd)


def pii_remove_help(input_file: str, entity_types=None):
    from pyrecdp.primitives.llmutils import pii_remove
    from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
    with SparkContext(input_file) as ctx:
        spark_df = ctx.ds
        model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")
        output_dataset = pii_remove(dataset=spark_df, text_column="text", model_root_path=model_root_path,
                                    show_secret_column=True, inplace=True, entity_types=entity_types)
        ctx.show(output_dataset)


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

    # def test_toxicity_score(self):
    #     from pyrecdp.primitives.llmutils import toxicity_score
    #     from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
    #     huggingface_config_path = os.path.join(RECDP_MODELS_CACHE, "models--xlm-roberta-base")
    #     file_path = os.path.join(cur_dir, "data/llm_data/tiny_c4_sample_for_pii.jsonl")
    #     save_path = os.path.join(cur_dir, "data/output/toxicity_score")
    #     toxicity_score(file_path, save_path, "jsonl", text_key='text',
    #                    threshold=0, model_type="multilingual", huggingface_config_path=huggingface_config_path)

    # def test_toxicity_score_spark(self):
    #     from pyrecdp.primitives.llmutils import toxicity_score_spark
    #     from pyrecdp.core import SparkDataProcessor
    #     from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
    #     huggingface_config_path = os.path.join(RECDP_MODELS_CACHE, "models--xlm-roberta-base")
    #     data_file = f'file://{os.path.join(cur_dir, "data/llm_data/tiny_c4_sample_for_pii.jsonl")}'
    #     rdp = SparkDataProcessor()
    #     spark = rdp.spark
    #     spark_df = spark.read.json(data_file)
    #     toxicity_score_df = toxicity_score_spark(spark_df, text_key='text',
    #                                              threshold=0, model_type="multilingual",
    #                                              huggingface_config_path=huggingface_config_path)
    #     toxicity_score_df.show()

    def test_diversity_analysis(self):
        from pyrecdp.primitives.llmutils import diversity_indicate
        data_dir = "tests/data/llm_data/arxiv_sample_100.jsonl"
        output_path = "tests/data/output/diversity_out_arxiv"
        in_type = "jsonl"
        diversity_indicate(data_dir, in_type, output_path)

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

    #TODO: Failed after using add-on requirement, fix later
    def test_pii_remove_spark(self):
        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl")

    def test_pii_remove_email_spark(self):
        from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType
        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl", entity_types=[PIIEntityType.EMAIL])

    def test_pii_remove_phone_spark(self):
        from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType
        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl", entity_types=[PIIEntityType.PHONE_NUMBER])

    def test_pii_remove_name_spark(self):
        from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType
        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl", entity_types=[PIIEntityType.NAME])

    def test_pii_remove_password_spark(self):
        from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType
        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl", entity_types=[PIIEntityType.PASSWORD])

    def test_pii_remove_ip_spark(self):
        from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType
        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl", entity_types=[PIIEntityType.IP_ADDRESS])

    def test_pii_remove_key_spark(self):
        from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType

        pii_remove_help("tests/data/llm_data/tiny_c4_sample_for_pii.jsonl", entity_types=[PIIEntityType.KEY])

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
                'South Dakota.\n\nThe plant slaughters 19,500 pigs a day — 5 '
                'percent of U.S. pork.')]
            spark_df = ctx.spark.createDataFrame(pd.DataFrame(samples, columns=["text", "target"]))
            ret_df = sentence_split(spark_df)
            ctx.show(ret_df)
            for _, row in ret_df.toPandas().iterrows():
                self.assertEqual(row["text"], row["target"])

    # def test_pdf_to_json(self):
    #     from pyrecdp.primitives.llmutils.document_extractor import pdf_to_text
    #     in_file = "tests/data/llm_data/document/layout-parser-paper.pdf"
    #     out_file = "tests/data/llm_data/document/layout-parser-paper.pdf.jsonl"
    #     pdf_to_text(in_file, out_file)

    # def test_docx_to_json(self):
    #     from pyrecdp.primitives.llmutils.document_extractor import docx_to_text
    #     in_file = "tests/data/llm_data/document/handbook-872p.docx"
    #     out_file = "tests/data/output/document/handbook-872p.docx.jsonl"
    #     docx_to_text(in_file, out_file)

    # def test_image_to_json(self):
    #     from pyrecdp.primitives.llmutils.document_extractor import image_to_text
    #     in_file = "tests/data/llm_data/document/layout-parser-paper-10p.jpg"
    #     out_file = "tests/data/output/document/layout-parser-paper-10p.jpg.jsonl"
    #     image_to_text(in_file, out_file)

    # def test_document_to_json(self):
    #     from pyrecdp.primitives.llmutils.document_extractor import document_to_text
    #     input_dir = "tests/data/llm_data/document"
    #     out_file = "tests/data/output/document/document.jsonl"
    #     document_to_text(input_dir, out_file, use_multithreading=True)

    def test_text_pipeline_optimize_with_one_config_file(self):
        from pyrecdp.primitives.llmutils.pipeline_hpo import text_pipeline_optimize
        input_pipeline_hpo_file = "tests/config/llm/pipeline/pipeline_hpo.yaml.template"
        output_pipeline_file = "tests/config/llm/pipeline/pipeline_hpo.yaml"
        text_pipeline_optimize(input_pipeline_hpo_file, output_pipeline_file)

    def test_text_pipeline_optimize_with_separate_config_file(self):
        from pyrecdp.primitives.llmutils.pipeline_hpo import text_pipeline_optimize
        input_pipeline_file = "tests/config/llm/pipeline/pipeline.yaml.template"
        output_pipeline_file = "tests/config/llm/pipeline/pipeline.yaml"
        input_hpo_file = "tests/config/llm/pipeline/hpo.yaml"
        text_pipeline_optimize(input_pipeline_file, output_pipeline_file, input_hpo_file)
