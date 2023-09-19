# RecDP Data LLMUtils

RecDP Data LLMUtils is a set of python components that enables quick and easy establish of your own LLM data preparation pipeline.
Phase 1 - By Sep

| Type                                           | Description                                               | sub tasks options provided           | Verified dataset & size               |
| ---------------------------------------------- | --------------------------------------------------------- | ------------------------------------ | ------------------------------------- |
| [ Conveter ]( #Conveter )                      | Convert jsonl, csv to parquet                             | jsonl, csv                           | RedPajama - 2 TB                      |
| [ Filter ]( #Filter )                          | Filter out document based on condition                    | profanity, black-list, url, length   | RedPajama - 2 TB                      |
| [ Language Identify ]( #Language_identify )    | Inentify major language type of document                  | en, zh, fr, de, .. total 25 langs    | RedPajama - 2 TB                      |
| [ Classify ]( #Classify )                      | Classify data into sub buckets                            | meta, language                       | RefinedWeb - 1.7 TB                   |
| [ Fuzzy Deduplicator ]( #Fuzzy_deduplication ) | Detect and reduce duplication based on document context   | minHashLSH, minHashLSH-shortdoc      | PILE - 200 GB                         |
| [ Decontamination ](#Decontamination )         | Detect and reduce duplication based on exact same content | sha256-based                         | RefinedWeb - 1.7 TB, RedPajama - 2 TB |
| [ PII Removal ]( #PII-Removal )                | Detect and replace personal infomation in document        | email, phone, ip, username, password | RefinedWeb - 1.7 TB                   |
| [ Text Normalization ]( #Text_normalization )  | Fix and clean texts                                       | fyfy, punctuation_normalization      | RedPajama - 2 TB , RedPajama - 2 TB   |
| [ Text Fixer ]( #Text_Fixer )                  | Clean repeated format in html, latex, codes               | html, latex, codes                   | RefinedWeb - 1.7 TB                   |
| [ Document splitter  ]( #Document_Splitter )   | Split Document into multiple sub documents                | chapter based, length based          | RefinedWeb - 1.7 TB                   |


Phase 2 - By Oct

| Type                                                        | Description                            |
| ----------------------------------------------------------- | -------------------------------------- |
| [ Prompt Enhancement ]( #Prompt_Enhancement )               | Improve Prompt quality                 |
| [ Data Augmentation ]( #Data_Augmentation )                 | Edits and transforms samples           |
| [ Text Quality Assesement ]( #Text_Quality_Assesement )     | Selects top samples based on ranking   |
| [ Toxicity and Bias Control ]( #Toxicity_and_Bias_Control ) | DeItects and removes duplicate samples |


Phase 3 - By Nov


# Getting Start

## Deploy
```
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre
pip install pyrecdp --pre
```

## Run component

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

### Prompt Enhancement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Data Augmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Text Quality Assesement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 

### Toxicity and Bias Control [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

Put some text to show expected input and output and how it validated 