# RecDP LLM - LLM data preparation utility

RecDP LLM is a set of python components that enables quick and easy establish of your own LLM data preparation pipeline.
* 10 general LLM data process components for Foundation model and finetune model training.
* 4 LLM data quality enhancement module for finetune model training
* 2 use cases for foundation model data prepration and finetune model data preparation.

## General - Foundation & FineTune

| Type                                                                                                                       | notebook                                                                                                                                                                                                   | Description                                               | supports                                             | Verified dataset & size               |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- |
| [ DocumentExtract ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/document_extractor.py)                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/document_extract.ipynb)            | extract text from unstructured format                          | jpg, png, pdf, docx,                                 | RefinedWeb - 1.7 TB                   |
| [ Reader ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_reader.py#L16)                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/reader.ipynb)            | Read data from directory                            | jsonl, parquet,                                 | RefinedWeb - 1.7 TB                   |
| [ Converter ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_converter.py)                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/convert.ipynb)            | Read and convert unstructed data to unified format                           | html, document, image, pdf, ...                                 | RefinedWeb - 1.7 TB                   |
| [ Filter ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/filter.py)                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/filter.ipynb)              | Filter out document based on condition                    | profanity-based, black-list, url_based, length_based | RedPajama - 2 TB                      |
| [ Text Bytesize ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_bytesize.py)                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/bytesize.ipynb)              | Get text bytes size                    |  | RedPajama - 2 TB                      |
| [ Text Fixer ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_fixer.py)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_fixer.ipynb)          | Clean repeated format in html, latex, codes               | html, latex, codes                                   | RefinedWeb - 1.7 TB                   |
| [ Language Identify ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_language_identify.py)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/language_identify.ipynb)   | Inentify major language type of document                  | en, zh, fr, de, .. total 25 langs                    | RedPajama - 2 TB                      |
| [ Fuzzy Deduplicator ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_deduplication.py#L99)           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/fuzzy_deduplication.ipynb) | Detect and reduce duplication based on document context   | minHashLSH                     | PILE - 200 GB                         |
| [ Global Decuplicator ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_deduplication.py#L194)         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/decontamination.ipynb)     | Detect and reduce duplication based on exact same content | sha256-hash                                          | RefinedWeb - 1.7 TB, RedPajama - 2 TB |
| [ Rouge Score Decuplicator ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_compare_dedup.py#L157)         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/rouge_score_deduplication.ipynb)     | Remove similar data by calculating the rough score                                         | alpaca |
| [ Repetition Removal ](#)         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/repetition_remove.ipynb)     | Detect and reduce repetition context in same document |                                           | RefinedWeb - 1.7 TB, RedPajama - 2 TB |
| [ Document splitter  ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_split.py)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/document_split.ipynb)      | Split Document into multiple sub documents                | chapter_based, length_based                          | RefinedWeb - 1.7 TB                   |
| [ PII Removal ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_pii_remove.py)                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/pii_removal.ipynb)         | Detect and replace personal infomation in document        | email, phone, ip, username, password                 | RefinedWeb - 1.7 TB                   |
| [ User Defined Transform ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_custom.py)                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/custom_map.ipynb)         | Easy way to plugin user defined map function        | parallel with ray or spark                 | RefinedWeb - 1.7 TB                   |
| [ User Defined Filter ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_custom.py)                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/custom_filter.ipynb)         | Easy way to plugin user defined filter function        | parallel with ray or spark                  | RefinedWeb - 1.7 TB                   |
| [ Writer ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_writer.py#L7)                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/writer.ipynb)            | write data to directory                            | jsonl, parquet                               | RefinedWeb - 1.7 TB                   |
| [ ClassifyWriter ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/operations/text_writer.py#L47)                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/classify.ipynb)            | Classify and write data into sub buckets                            | meta fields, language                                | RefinedWeb - 1.7 TB                   |
| [ Prompt Enhancement ](#)                                                                                                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/prompt_enhancement.ipynb)      | creates high-complexity instructions from existing instruct-tuned LLM models       | PromptSource, self-instruct, evol-instruct(wizardLM) | alpaca |
| [ Tokenization ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/tokenize_and_save/)                                                                                                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)      | using LLAMA2 tokenizer and save as Megatron    | LLAMA2 tokenizer | RefinedWeb - 1.7 TB |

## LLM Data Quality Analysis

| Diversity   |  GPT-3 Scoring | Toxicity | Perplexity |
| :-------- | :---------- | :------------|:------------|
| Visualize the diversity distribution of data | Leverage GPT-3 to scoring | Visualize Toxicity probability |  Visualize Perplexity Distribution |
| ![diversity](/RecDP/resources/diversity_analysis.png) | ![quality](/RecDP/resources/quality_scoring.png) | ![toxicity](/RecDP/resources/toxicity_analysis.png)| ![perxicity](/RecDP/resources/perplexity.png) |
| [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/data_diversity_control.ipynb) | [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_quality_assessment.ipynb) | [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/toxicity_bias_control.ipynb) | [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_perplexity.ipynb) |

## Getting Start

### Deploy
```
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre graphviz
pip install pyrecdp[LLM] --pre
```

### Data pipeline

#### 1. RAG Data Pipeline - Build from public HTML [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/rag_pipeline.ipynb)
Retrieval-augmented generation (RAG) for large language models (LLMs) aims to improve prediction quality by using an external datastore at inference time to build a richer prompt that includes some combination of context, history, and recent/relevant knowledge (RAG LLMs).
Recdp LLM can provide a pipeline for ingesting data from a source and indexing it. We mainly provide the following capabilities.
- **Load Data**: Load your data from source. You can use `UrlLoader` or `DirectoryLoader` for this.
- **Improve Data Quality**: Clean up text for LLM RAG to use. It mainly solves the problem of sentences being split by incorrect line breaks after parsing the file, removing special characters, fixing unicode errors,  and so on.
- **Split Text**: `DocumentSplit` helps break large Documents into smaller chunks. This is useful for indexing data and make it better used by the model.
- **Vector Store**: In order to retrieve your data, We provide `DocumentIngestion` use a VectorStore and Embeddings model to store and index your data.

Here is a basic RAG Data Pipeline example:
```python
from pyrecdp.primitives.operations import *
from pyrecdp.LLM import TextPipeline

pipeline = TextPipeline()
ops = [
    UrlLoader(urls=["https://www.intc.com/news-events/press-releases/detail/1655/intel-reports-third-quarter-2023-financial-results"], max_depth=0, target_tag='div', target_attrs={'class': 'main-content'}),
    DirectoryLoader(files_path, glob="**/*.pdf"),
    RAGTextFix(),
    DocumentSplit(),
    DocumentIngestion(
        vector_store='FAISS',
        vector_store_args={
            "output_dir": "ResumableTextPipeline_output",
            "index": "test_index"
        },
        embeddings='HuggingFaceEmbeddings',
        embeddings_args={
            'model_name': f"{model_root_path}/sentence-transformers/all-mpnet-base-v2"
        }
    ),
]
pipeline.add_operations(ops)
pipeline.execute()
```

#### 2. Finetune Data Pipeline - Build finetune dataset from Plain Text to QA

```
from pyrecdp.LLM import TextPipeline
from pyrecdp.primitives.operations import ParquetReader, TextPrompt, TextToQA, ParquetWriter

text_key = "text_prompt"
pipeline = TextPipeline()
ops = [
    ParquetReader(dataset_path),
    TextPrompt(dataset_name="text", prompt_name="generate_qa",new_name=text_key),
    TextToQA(outdir=result_path,text_key=text_key),
    ParquetWriter(result_path)
]
pipeline.add_operations(ops)
pipeline.execute()
```

#### 3. Finetune Data Pipeline - Downsize public finetune dataset

```
from pyrecdp.LLM import TextPipeline, ResumableTextPipeline
from pyrecdp.primitives.operations import *

import os
pipeline = ResumableTextPipeline()
pipeline.enable_statistics()
ops = [
    JsonlReader("{path-to-e2eAIOK}/RecDP/tests/data/alpaca/alpaca_data_50.jsonl"),
    TextPrompt(dataset_name="alpaca", prompt_name="causal_llm_1"),
    RandomSelect(fraction=0.3),
    TextToxicity(),
    TextDiversityIndicate(out_dir=out_dir, language="en", first_sent=False),
    TextQualityScorer(model="gpt3"),
    RougeScoreDedup(max_ratio=0.7, batch_size=10,score_store_path=os.path.join(out_dir,'RougeScorefiltered.parquet')),
    ParquetWriter("ResumableTextPipeline_output")
]
pipeline.add_operations(ops)
pipeline.execute()
```

#### 4. AutoHPO

Low-Code configuration with automated operators parameter tuning, allowing user to transform their own raw data toward a high quality dataset with low-effort. We coupled data processing with Quality Analisys as evaluation metrics, which will estimate data's quality before actual model finetuning/inference.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/pipeline_hpo.ipynb)
![image](https://github.com/chaojun-zhang/e2eAIOK/assets/4355494/f20cbde4-7221-4459-93a9-7f16af23377a)

```
from pyrecdp.primitives.llmutils.pipeline_hpo import text_pipeline_optimize

# input data path is configured in input_pipeline_file
input_pipeline_file = "config/pipeline_hpo.yaml.template"
input_hpo_file = 'config/hpo.yaml'
output_pipeline_file = "config/pipeline.yaml"

text_pipeline_optimize(input_pipeline_file, output_pipeline_file, input_hpo_file)
```

## * run with individual component
  * cmdline mode 
```
python pyrecdp/primitives/llmutils/language_identify.py \
    --data_dir tests/llm_data \
    --language_identify_output_dir output \
    --fasttext_model_dir ./cache/RecDP/models/lib.bin

```
  * operation-based API - ray mode
```
from pyrecdp.primitives.operations import LengthFilter
 
dataset = … # Ray Dataset
op = LengthFilter()
op.process_rayds(dataset)
```

  * operation-based API - spark mode
```
from pyrecdp.primitives.operations import LengthFilter

sparkdf = … # Spark Dataframe
op = LengthFilter()
op.process_spark(sparkdf)
```
