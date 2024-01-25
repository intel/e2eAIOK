import unittest
import sys
import pandas as pd
from pathlib import Path
import os
from IPython.display import display
from pyspark.sql import DataFrame

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
        if os.path.exists(output_path):
            try:
                dir_name_list = [i for i in os.listdir(output_path) if i.endswith('jsonl')]
                for dir_name in dir_name_list:
                    print(dir_name)
                    tmp_df = pd.read_parquet(os.path.join(output_path, dir_name))
                    print(f"total num_samples is {len(tmp_df)}")
                    display(tmp_df.head())
            except Exception as e:
                print(e)
            try:
                shutil.rmtree(output_path)
            except:
                pass
        return super().tearDown()

    def test_ResumableTextPipeline(self):
        pipeline = ResumableTextPipeline()
        ops = [
            JsonlReader("tests/data/llm_data/"),
            LengthFilter(),
            ProfanityFilter(),
            LanguageIdentify(fasttext_model_dir=os.path.join(RECDP_MODELS_CACHE, "lid.bin")),
            PerfileParquetWriter("ResumableTextPipeline_output")
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_ResumableTextPipeline_import(self):
        pipeline = ResumableTextPipeline(pipeline_file='tests/data/import_test_pipeline.json')
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

    # def test_ResumableTextPipeline_customer_reload_function(self):
    #     pipeline = ResumableTextPipeline(pipeline_file = "tests/data/custom_op_pipeline.json")
    #     pipeline.execute()
    #     del pipeline

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
        # pipeline.enable_statistics()
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
            LanguageIdentify(fasttext_model_dir=os.path.join(RECDP_MODELS_CACHE, "lid.bin"))
        ]
        pipeline.add_operations(ops)
        df = pd.read_parquet("tests/data/PILE/NIH_sample.parquet")
        ret = pipeline.execute(df)
        display(ret.to_pandas())
        del pipeline

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
        del pipeline

    def test_llm_rag_url_pipeline(self):
        model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")
        model_name = f"{model_root_path}/sentence-transformers/all-mpnet-base-v2"
        faiss_output_dir = 'tests/data/faiss'
        pipeline = TextPipeline()
        ops = [
            UrlLoader(["https://www.intc.com/news-events/press-releases/detail/"
                            "1655/intel-reports-third-quarter-2023-financial-results"]),
            DocumentSplit(text_splitter='RecursiveCharacterTextSplitter'),
            DocumentIngestion(
                vector_store='FAISS',
                vector_store_args={
                    "output_dir": faiss_output_dir,
                    "index": "test_index"
                },
                embeddings='HuggingFaceEmbeddings',
                embeddings_args={'model_name': model_name}
            ),
        ]
        pipeline.add_operations(ops)
        pipeline.execute()
        del pipeline

    def test_llm_rag_url_pdf_pipeline(self):
        pipeline = TextPipeline()
        ops = [
            UrlLoader(["https://www.intc.com/news-events/press-releases/detail/"
                            "1655/intel-reports-third-quarter-2023-financial-results"],
                      max_depth=2),
            DirectoryLoader("tests/data/press_pdf", glob="**/*.pdf"),
            RAGTextFix(),
            DocumentSplit(text_splitter='RecursiveCharacterTextSplitter')
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        display(ret.to_pandas())
        del pipeline

    def test_llm_rag_pdf_return_db_pipeline(self):
        model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")
        model_name = f"{model_root_path}/sentence-transformers/all-mpnet-base-v2"
        faiss_output_dir = 'tests/data/faiss'
        pipeline = TextPipeline()
        ops = [
            DirectoryLoader("tests/data/press_pdf", glob="**/*.pdf"),
            DocumentSplit(text_splitter='RecursiveCharacterTextSplitter'),
            DocumentIngestion(
                vector_store='FAISS',
                vector_store_args={
                    "output_dir": faiss_output_dir,
                    "index": "test_index"
                },
                embeddings='HuggingFaceEmbeddings',
                embeddings_args={'model_name': model_name},
                return_db_handler=True
            ),
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        display(ret)
        del pipeline

    def test_llm_rag_pdf_use_existing_db_pipeline(self):
        from pyrecdp.core.import_utils import import_sentence_transformers, check_availability_and_install
        check_availability_and_install(["langchain", "faiss-cpu"])

        # Present that someone else already define the handler ##
        model_root_path = os.path.join(RECDP_MODELS_CACHE, "huggingface")
        model_name = f"{model_root_path}/sentence-transformers/all-mpnet-base-v2"
        faiss_output_dir = 'tests/data/faiss'
        embeddings='HuggingFaceEmbeddings'
        embeddings_args={'model_name': model_name}

        import_sentence_transformers()
        from pyrecdp.core.class_utils import new_instance
        from langchain.vectorstores.faiss import FAISS
        embeddings = new_instance('langchain.embeddings', embeddings, **embeddings_args)
        db = FAISS.load_local(faiss_output_dir, embeddings, 'test_index')

        pipeline = TextPipeline()
        ops = [
            DirectoryLoader("tests/data/press_pdf", glob="**/*.pdf"),
            DocumentSplit(text_splitter='RecursiveCharacterTextSplitter'),
            DocumentIngestion(
                vector_store='FAISS',
                db_handler = db,
                return_db_handler = True,
                embeddings='HuggingFaceEmbeddings',
                embeddings_args={'model_name': model_name},
            ),
        ]
        pipeline.add_operations(ops)
        ret = pipeline.execute()
        display(ret)
        del pipeline

    def test_llm_rag_pipeline_cnvrg(self):
        from pyrecdp.primitives.operations import UrlLoader,RAGTextFix,CustomerDocumentSplit,TextCustomerFilter,JsonlWriter
        from pyrecdp.LLM import TextPipeline

        def prepare_nltk_model(model, lang):
            import nltk
            nltk.download('punkt')

        from pyrecdp.core.model_utils import prepare_model
        prepare_model(model_type="nltk", model_key="nltk_rag_cnvrg", prepare_model_func=prepare_nltk_model)
        urls = ['https://app.cnvrg.io/docs/',
                'https://app.cnvrg.io/docs/core_concepts/python_sdk_v2.html',
                'https://app.cnvrg.io/docs/cli_v2/cnvrgv2_cli.html',
                'https://app.cnvrg.io/docs/collections/tutorials.html']

        def custom_filter(text):
            from nltk.tokenize import word_tokenize
            ret_txt = None
            if len(word_tokenize(text)) >10:
                if text.split(' ')[0].lower()!='version':
                    ret_txt = text
            return ret_txt != None

        def chunk_doc(text,max_num_of_words):
            from nltk.tokenize import word_tokenize,sent_tokenize
            text= text.strip()
            if len(word_tokenize(text)) <= max_num_of_words:
                return [text]
            else:
                chunks = []
                # split by sentence
                sentences = sent_tokenize(text)
                # print('number of sentences: ', len(sentences))
                words_count = 0
                temp_chunk = ""
                for s in sentences:
                    temp_chunk+=(s+" ")
                    words_count += len(word_tokenize(s))
                    if len(word_tokenize(temp_chunk))> max_num_of_words:
                        chunks.append(temp_chunk)
                        words_count = 0
                        temp_chunk = ""

                return chunks

        pipeline = TextPipeline()
        ops = [
            UrlLoader(urls, max_depth=2),
            RAGTextFix(str_to_replace={'\n###': '', '\n##': '', '\n#': ''}, remove_extra_whitespace=True),
            CustomerDocumentSplit(func=lambda text: text.split('# ')[1:]),
            TextCustomerFilter(custom_filter),
            CustomerDocumentSplit(func=chunk_doc, max_num_of_words=50),
            GlobalDeduplicate(),
            JsonlWriter("TextPipeline_output_jsonl")
        ]
        pipeline.add_operations(ops)
        ds:DataFrame = pipeline.execute()
        display(ds.toPandas())
        del pipeline