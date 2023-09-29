# RecDP LLM - LLM data preparation utility

RecDP LLM is a set of python components that enables quick and easy establish of your own LLM data preparation pipeline.
* 10 general LLM data process components for Foundation model and finetune model training.
* 4 LLM data quality enhancement module for finetune model training
* 2 use cases for foundation model data prepration and finetune model data preparation.

## General - Foundation & FineTune

| Type                                                                                                                       | notebook                                                                                                                                                                                                   | Description                                               | supports                                             | Verified dataset & size               |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- |
| [ Convert ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/convert.py)                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/convert.ipynb)             | Convert individual document, jsonl, csv to parquet        | text, jsonl, csv                                     | RedPajama - 2 TB                      |
| [ Filter ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/filter.py)                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/filter.ipynb)              | Filter out document based on condition                    | profanity_check, black-list, url_based, length_based | RedPajama - 2 TB                      |
| [ Language Identify ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/language_identify.py)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/language_identify.ipynb)   | Inentify major language type of document                  | en, zh, fr, de, .. total 25 langs                    | RedPajama - 2 TB                      |
| [ Classify ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/convert.py)                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/classify.ipynb)            | Classify data into sub buckets                            | meta fields, language                                | RefinedWeb - 1.7 TB                   |
| [ Fuzzy Deduplicator ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/classify.py)           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/fuzzy_deduplication.ipynb) | Detect and reduce duplication based on document context   | minHashLSH, minHashLSH-shortdoc                      | PILE - 200 GB                         |
| [ Decontamination ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/decontaminate.py)         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/decontamination.ipynb)     | Detect and reduce duplication based on exact same content | sha256-hash                                          | RefinedWeb - 1.7 TB, RedPajama - 2 TB |
| [ PII Removal ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/pii_remove.py)                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/pii_removal.ipynb)         | Detect and replace personal infomation in document        | email, phone, ip, username, password                 | RefinedWeb - 1.7 TB                   |
| [ Text Normalization ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/text_normalization.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_normalization.ipynb)  | Fix and clean texts                                       | ftfy, punctuation_normalization                      | RedPajama - 2 TB , RedPajama - 2 TB   |
| [ Text Fixer ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/text_fixer.py)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_fixer.ipynb)          | Clean repeated format in html, latex, codes               | html, latex, codes                                   | RefinedWeb - 1.7 TB                   |
| [ Document splitter  ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/sentence_split.py)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/document_split.ipynb)      | Split Document into multiple sub documents                | chapter_based, length_based                          | RefinedWeb - 1.7 TB                   |

## LLM data quality enhancement module

| Type                                                                                                                            | notebook                                                                                                                                                                                                       | Description                                                                        | supports                                             |
| :------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :--------------------------------------------------- |
| [ Prompt Enhancement ](#)                                                                                                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/prompt_enhancement.ipynb)      | creates high-complexity instructions from existing instruct-tuned LLM models       | PromptSource, self-instruct, evol-instruct(wizardLM) |
| [ Text Quality Assessment ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/quality_classifier.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_quality_assessment.ipynb) | Assess text quality with a logistic regression classifier                          | GPT-3 quality scorer,                                |
| [ Data Diversity Control ](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/primitives/llmutils/diversity_analysis.py)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/data_diversity_control.ipynb)  | Control data's diversity and coverage                                              | rouge-l similarity                                   |
| [ Toxicity and Bias Control ](#)                                                                                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/toxicity_bias_control.ipynb)   | Calculates toxicity scores for text objects and filters according to the threshold | TBD                                                  |


# Getting Start

## Deploy
```
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre
pip install pyrecdp --pre
```

## * run with pipeline
```
from pyrecdp.primitives.operations import *
from pyrecdp.LLM import TextPipeline
from IPython.display import display

pipeline = TextPipeline()
ops = [
    JsonlReader(input_dir = "in_path"),
    URLFilter(),
    LengthFilter(),
    ProfanityFilter(),
    TextFix(),
    LanguageIdentify(),
    Classify(),
    FuzzyDeduplicate(),
    DocumentSplit(),
    GlobalDeduplicate(),
    PIIRemoval(),
    ParquetWriter(out_dir = "out_path"),
]
ret = pipeline.add_operations(ops).execute(dataset)

pd = ret.preview_as_pandas()
display(pd)

pipeline.export_to_yaml("custom_llm_data_pipeline.yaml")
```

## * run with individual component
  * check colab links of individual component above
  * cmdline mode provided to process your data from data_dir
  * python function mode provided to easily integrate to your codes
  * spark dataframe mode provided to easily integrate to your spark env


## LLM Data Quality Analysis

| Diversity   |  GPT-3 Scoring | Toxicity | 
| :-------- | :---------- | :------------|
| Visualize the diversity distribution of data | Leverage GPT-3 to scoring | Visualize Toxicity probability |
| ![diversity](/RecDP/resources/diversity_analysis.png) | ![quality](/RecDP/resources/quality_scoring.png) | ![toxicity](/RecDP/resources/toxicity_analysis.png)|
| [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_quality_assessment.ipynb) | [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/data_diversity_control.ipynb) | [learn more](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/toxicity_bias_control.ipynb) |



## Use Cases

### * Foundation Model data preparation
```
from pyrecdp.LLM import TextPipeline
pipeline = TextPipeline("usecase/itl_foundation_pipeline.yaml")
ret = pipeline.execute()
pd = ret.preview_as_pandas()
display(pd)
```

### * FineTune Model Data Preparation
```
from pyrecdp.LLM import TextPipeline
pipeline = TextPipeline("usecase/itl_finetune_pipeline.yaml")
ret = pipeline.execute()
pd = ret.preview_as_pandas()
display(pd)
```



