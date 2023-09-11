import unittest
import sys
import pandas as pd
from pathlib import Path
import os
import wget
import urllib.error
pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)

from pyrecdp.primitives.llmutils import near_dedup, shrink_document_MP, text_to_jsonl_MP, pii_remove
from pyrecdp.primitives.llmutils import near_dedup, shrink_document_MP, text_to_jsonl_MP, filter_by_blocklist
from pyrecdp.primitives.llmutils import near_dedup, shrink_document_MP, text_to_jsonl_MP, language_identify, Classifier

cur_dir = str(Path(__file__).parent.resolve())


class Test_LLMUtils(unittest.TestCase):
    def setUp(self):
        self.data_files = ["tests/data/llm_data/NIH_sample.jsonl"]
        self.data_dir = "tests/data/llm_data/"
        self.dup_dir = "./near_dedup/"
        self.fasttext_model = "./fasttext_model/lid.bin"
        self.fasttext_mode_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def test_near_dedup(self):
        data_files = self.data_files
        dup_dir = self.dup_dir
        ngram_size = 13
        num_perm = 256
        bands = 9
        ranges = 13
        near_dedup(data_files, dup_dir, ngram_size, num_perm, bands, ranges)

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
        from dataclasses import dataclass

        @dataclass
        class PiiDetectRedactOption:
            path: str = "json"
            data_files: str = "tests/data/llm_data/arxiv_sample_100.jsonl"
            split: str = "train"
            text_column: str = "text"
            batch_size: int = 100
            num_proc: int = 8
            seed: int = 10
            save_path: str = "./pii_remove"
            save_format: str = "json"

        args = PiiDetectRedactOption()
        pii_remove(args)

    def test_filter_jsonl(self):
        data_dir = "tests/data/llm_data"
        out_dir = "tests/data/filter_out"
        filter_by_blocklist(data_dir, out_dir)

    def test_language_identify(self):
        data_files = self.data_files
        data_dir = self.data_dir
        fasttext_model_dir = os.path.abspath(self.fasttext_model)
        if not os.path.exists(fasttext_model_dir):
            os.makedirs(os.path.dirname(fasttext_model_dir), exist_ok=True)
            try:
                wget.download(self.fasttext_mode_url , out=fasttext_model_dir)
            except urllib.error.HTTPError:
                print("Faild to download DL languafe model. Please check your network.")
                exit(1)
        model = Path(fasttext_model_dir)
        classifier = Classifier(model, 'text', 'lang')
        language_identify_output_dir = os.path.join(data_dir, "language_identify")
        language_identify(data_files, classifier, language_identify_output_dir, enable_ray=False)