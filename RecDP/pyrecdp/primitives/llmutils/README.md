# RecDP LLM - LLM data preparation utility

RecDP LLM is a set of python components that enables quick and easy establish of your own LLM data preparation pipeline.
* 10 general LLM data process components for Foundation model and finetune model training.
* 4 LLM data quality enhancement module for finetune model training
* 2 use cases for foundation model data prepration and finetune model data preparation.

## General - Foundation & FineTune

| Type                                           | Description                                               | supports                                             | Verified dataset & size               |
| ---------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- |
| [ Convert ]( #Convert )                        | Convert individual document, jsonl, csv to parquet        | text, jsonl, csv                                     | RedPajama - 2 TB                      |
| [ Filter ]( #Filter )                          | Filter out document based on condition                    | profanity_check, black-list, url_based, length_based | RedPajama - 2 TB                      |
| [ Language Identify ]( #Language_identify )    | Inentify major language type of document                  | en, zh, fr, de, .. total 25 langs                    | RedPajama - 2 TB                      |
| [ Classify ]( #Classify )                      | Classify data into sub buckets                            | meta fields, language                                | RefinedWeb - 1.7 TB                   |
| [ Fuzzy Deduplicator ]( #Fuzzy_deduplication ) | Detect and reduce duplication based on document context   | minHashLSH, minHashLSH-shortdoc                      | PILE - 200 GB                         |
| [ Decontamination ](#Decontamination )         | Detect and reduce duplication based on exact same content | sha256-hash                                          | RefinedWeb - 1.7 TB, RedPajama - 2 TB |
| [ PII Removal ]( #PII-Removal )                | Detect and replace personal infomation in document        | email, phone, ip, username, password                 | RefinedWeb - 1.7 TB                   |
| [ Text Normalization ]( #Text_normalization )  | Fix and clean texts                                       | ftfy, punctuation_normalization                      | RedPajama - 2 TB , RedPajama - 2 TB   |
| [ Text Fixer ]( #Text_Fixer )                  | Clean repeated format in html, latex, codes               | html, latex, codes                                   | RefinedWeb - 1.7 TB                   |
| [ Document splitter  ]( #Document_Splitter )   | Split Document into multiple sub documents                | chapter_based, length_based                          | RefinedWeb - 1.7 TB                   |

## LLM data quality enhancement

| Type                                                        | Description                                                                        | supports                                             |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [ Prompt Enhancement ]( #Prompt_Enhancement )               | creates high-complexity instructions from existing instruct-tuned LLM models       | PromptSource, self-instruct, evol-instruct(wizardLM) |
| [ Text Quality Assessment ]( #Text_Quality_Assesement )     | Assess text auality with a logistic regression classifier                          | GPT-3 quality scorer,                                |
| [ Data Diversity Control ]( #Data_Diversity_Control )       | Control data's diversity and coverage                                              | rouge-l similarity                                   |
| [ Toxicity and Bias Control ]( #Toxicity_and_Bias_Control ) | Calculates toxicity scores for text objects and filters according to the threshold | TBD                                                  |


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

### Filter [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Language Identify [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Classify [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Near Dedup (Fuzzy Deduplication) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Decontamination (Global Deduplication) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### PII removal [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

## For Fine Tune

### Prompt Enhancement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Data Augmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Text Quality Assesement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Toxicity and Bias Control [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated