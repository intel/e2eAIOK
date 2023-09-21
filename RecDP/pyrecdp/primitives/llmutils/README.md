# RecDP LLM - LLM data preparation utility

RecDP LLM is a set of python components that enables quick and easy establish of your own LLM data preparation pipeline.
* 10 general LLM data process components for Foundation model and finetune model training.
* 4 LLM data quality enhancement module for finetune model training
* 2 use cases for foundation model data prepration and finetune model data preparation.

## General - Foundation & FineTune

| Type                                                                                                                                                 | Description                                               | supports                                             | Verified dataset & size               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- |
| [ Convert ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/convert.ipynb)                        | Convert individual document, jsonl, csv to parquet        | text, jsonl, csv                                     | RedPajama - 2 TB                      |
| [ Filter ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/filter.ipynb)                          | Filter out document based on condition                    | profanity_check, black-list, url_based, length_based | RedPajama - 2 TB                      |
| [ Language Identify ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/language_identify.ipynb)    | Inentify major language type of document                  | en, zh, fr, de, .. total 25 langs                    | RedPajama - 2 TB                      |
| [ Classify ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/classify.ipynb)                      | Classify data into sub buckets                            | meta fields, language                                | RefinedWeb - 1.7 TB                   |
| [ Fuzzy Deduplicator ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/fuzzy_deduplication.ipynb) | Detect and reduce duplication based on document context   | minHashLSH, minHashLSH-shortdoc                      | PILE - 200 GB                         |
| [ Decontamination ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/decontamination.ipynb)        | Detect and reduce duplication based on exact same content | sha256-hash                                          | RefinedWeb - 1.7 TB, RedPajama - 2 TB |
| [ PII Removal ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/pii_removal.ipynb)                | Detect and replace personal infomation in document        | email, phone, ip, username, password                 | RefinedWeb - 1.7 TB                   |
| [ Text Normalization ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_normalization.ipynb)  | Fix and clean texts                                       | ftfy, punctuation_normalization                      | RedPajama - 2 TB , RedPajama - 2 TB   |
| [ Text Fixer ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_fixer.ipynb)                  | Clean repeated format in html, latex, codes               | html, latex, codes                                   | RefinedWeb - 1.7 TB                   |
| [ Document splitter  ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/document_split.ipynb)      | Split Document into multiple sub documents                | chapter_based, length_based                          | RefinedWeb - 1.7 TB                   |

## LLM data quality enhancement

| Type                                                                                                                                                          | Description                                                                        | supports                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------- | :--------------------------------------------------- |
| [ Prompt Enhancement ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/prompt_enhancement.ipynb)           | creates high-complexity instructions from existing instruct-tuned LLM models       | PromptSource, self-instruct, evol-instruct(wizardLM) |
| [ Text Quality Assessment ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_quality_assessment.ipynb) | Assess text quality with a logistic regression classifier                          | GPT-3 quality scorer,                                |
| [ Data Diversity Control ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/data_diversity_control.ipynb)   | Control data's diversity and coverage                                              | rouge-l similarity                                   |
| [ Toxicity and Bias Control ](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/toxicity_bias_control.ipynb) | Calculates toxicity scores for text objects and filters according to the threshold | TBD                                                  |


# Getting Start

## Deploy
```
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre
pip install pyrecdp --pre
```

# Use Cases

## * Foundation Model data preparation

## * FineTune Model Data Preparation

# LLM Components

## General

### Convert [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/convert.ipynb)
Convert types of format into parquet. We usually see jsonl, jsonl.zst, individual text files, wet, csv, etc are being used to store raw data in LLM domain. This component is used to convert types of raw files to parquet format with an align schema as below

| text                | meta                              | supports                                             |
| ------------------- | --------------------------------- | ---------------------------------------------------- |
| This is a cool tool | {'source': 'dummy', 'lang': 'en'} | PromptSource, self-instruct, evol-instruct(wizardLM) |
| llm is fun          | {'source': 'dummy', 'lang': 'en'} | GPT-3 quality scorer,                                |
| ...                 | {'source': 'dummy', 'lang': 'en'} | rouge-l similarity                                   |


### Filter [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/filter.ipynb)
Filter out document based on provided rules. 

### Language Identify [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/language_identify.ipynb)
Inentify major language type of document.

### Classify [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/classify.ipynb)
Classify data into sub buckets. Classify rule can be based on language, data source category, etc.
 

### Near Dedup (Fuzzy Deduplication) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/fuzzy_deduplication.ipynb)
Detect and reduce duplication based on document context. We use DataSketch minHash as the base algorithm to calculate (hash, band_id) pair for each documents, then we use spark for minHashLSH and use networkx to get all connected components, and convert connected components to a duplication list. Eventually we remove duplications based on the duplication list.


### Decontamination (Global Deduplication) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/decontamination.ipynb)
Detect and reduce duplication based on exact same content. We use sha256 to generate normalized hash value for each document. Then we use spark to generate duplication list. Eventually we use another spark function to remove duplicated documents.

### PII removal [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/pii_removal.ipynb)
Detect and replace personal infomation in document.

### Text Normalization [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_normalization.ipynb)
Fix and clean texts using ftfy and remove unnecessary punctuation to only keep the text meaning. 

### Text Fixer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_fixer.ipynb)
Clean repeated format in html, latex, codes. 

### Document Split [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/document_split.ipynb)
Split Document into multiple sub documents.

## For Fine Tune

### Prompt Enhancement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/prompt_enhancement.ipynb)
Creates high-complexity instructions from existing instruct-tuned LLM models.

### Text Quality Assesement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/text_quality_assessment.ipynb)
Assess text quality with a logistic regression classifier. 

### Data Diversity Control [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/data_diversity_control.ipynb)
Control data's diversity and coverage.

### Toxicity and Bias Control [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/RecDP/examples/notebooks/llmutils/toxicity_bias_control.ipynb)
Calculates toxicity scores for text objects and filters according to the threshold.