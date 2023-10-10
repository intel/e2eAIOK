# RecDP - one stop toolkit for AI data process

We provide intel optimized solution for

* [**Tabular**](pyrecdp/autofe/README.md) - Auto Feature Engineering Pipeline, 50+ essential primitives for feature engineering.
* [**LLM Text**](pyrecdp/LLM/README.md) - 10+ essential primitives for text clean, fixing, deduplication, 4 quality control module, 2 built-in high quality data pipelines.

## Getting Started

```
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre graphviz
pip install pyrecdp --pre
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

Low Code to build your own pipeline
```
from pyrecdp.LLM import ResumableTextPipeline
pipeline = ResumableTextPipeline("usecase/finetune_pipeline.yaml")
ret = pipeline.execute()
```
or
```
from pyrecdp.primitives.operations import *
from pyrecdp.LLM import ResumableTextPipeline

pipeline = ResumableTextPipeline()
ops = [
    JsonlReader("data/"),
    URLFilter(),
    LengthFilter(),
    ProfanityFilter(),
    TextFix(),
    LanguageIdentify(),
    PIIRemoval(),
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