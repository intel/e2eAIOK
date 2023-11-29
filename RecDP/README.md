# RecDP - one stop toolkit for AI data process

We provide intel optimized solution for

* [**Auto Feature Engineering**](pyrecdp/autofe/README.md) - Automatical Feature Engineering for Tabular input including 50+ essential primitives. Supporting numerical features, categorical features, text features, time series features, distributed process based on Spark. Feature Importance Analysis based on LightGBM.
* [**LLM Data Preparation**](pyrecdp/LLM/README.md) - 50+ essential operators for RAG data preparation, Finetune Data preparation, Foundation Model text prepration. Supporting text from PDF, words, crawl from URLs, distributed text process based on Ray or Spark, Quality Evaluation based on GPT-3, Divesity, Toxicity, Rouge-similarity, Perplexity.

## How it works

Install this tool through pip. 

```
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre graphviz
pip install pyrecdp[all] --pre
```

## RecDP - Tabular
[learn more](pyrecdp/autofe/README.md)

* Auto Feature Engineering Pipeline
![Auto Feature Engineering Pipeline](resources/autofe_pipeline.jpg)

Only **3** lines of codes to generate new features for your tabular data. Usually 5x new features can be found with up to 1.2x accuracy boost
```
from pyrecdp.autofe import AutoFE

pipeline = AutoFE(dataset=train_data, label=target_label, time_series = 'Day')
transformed_train_df = pipeline.fit_transform()
```

* High Performance on Terabyte Tabular data processing
![Performance](resources/recdp_performance.jpg)

## RecDP - LLM
[learn more](pyrecdp/LLM/README.md)

* Low-code Fault-tolerant Auto-scaling Parallel Pipeline
![LLM Pipeline](resources/llm_pipeline.jpg)

```
from pyrecdp.primitives.operations import *
from pyrecdp.LLM import ResumableTextPipeline

pipeline = ResumableTextPipeline()
ops = [
    Url_Loader(urls="your_url"),
    DocumentSplit(),
    ProfanityFilter(),
    PIIRemoval(),
    ...
    PerfileParquetWriter("ResumableTextPipeline_output")
]
pipeline.add_operations(ops)
pipeline.execute()
```

## LICENSE
* Apache 2.0

## Dependency
* Spark 3.4.*
* python 3.*
* Ray 2.7.*