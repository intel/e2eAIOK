# RecDP

# INTRODUCTION
RecDP is a Data Process python module, specifically designed for Recommender System.
* Easy-to-use – Wrap often used operations with simple APIs.
* Collaborative pipeline with spark - provide stableness and scalability of handling huge dataset with spark as underlying distributed data process engine.
* Optimized Performance - 1) Adaptive dataframe plan decision making; 2) Intel-OAP accelerator extensions (SIMD, Cache, Native).
* Feature Engineer oriented – advanced feature engineering functions (target encoding) 

# Getting Start
## install with pip (require preinstall spark)
```
# default version is working with spark 3.1
pip install pyrecdp
```

## use docker to setup pyrecdp
```
python3 scripts/start_e2eaiok_docker.py
sshpass -p docker ssh sr414 -p 12349
pip install pyrecdp
```

## examples
[More examples](tests/)

categorify a source data
* convert 'language' column from 'text' to 'unique_integral_id'
* codes link: [tests/test_categorify.py](tests/test_categorify.py)
``` python
from pyrecdp.data_processor import *
from pyrecdp.utils import *
proc = DataProcessor(spark, path_prefix, cur_folder)
proc.reset_ops([Categorify(['language'])])
df = proc.transform(df)
```

sort a list by frequency
* when each cell data is a list, organize this list to unique_value order by frequency
* codes link: [tests/test_sortArrayByFrequency.py](tests/test_sortArrayByFrequency.py)
``` python
from pyrecdp.data_processor import *
from pyrecdp.utils import *
proc = DataProcessor(spark, path_prefix, cur_folder)
# group langugage by hour of day
df = df.groupby('dt_hour').agg(f.collect_list("language").alias("language_list"))
# to sort language by its showing frequency in this hour
df = df.withColumn("sorted_langugage", f.expr(f"sortStringArrayByFrequency(language_list)"))
df = proc.transform(df)
```
![image](https://user-images.githubusercontent.com/4355494/144941079-9a06fc88-38fe-454a-b33a-b559edc636de.png)


## use cases
* Recsys2021 example [url](https://github.com/oap-project/recdp/blob/master/examples/notebooks/recsys2021/final_submission_feature_engineering.ipynb)
* Recsys2020 example [url](https://github.com/oap-project/recdp/blob/master/examples/notebooks/recsys2020/recsys2020_feature_engineering.ipynb)
* Recsys2020 multiitem-categorify example(support for Analytics Zoo Friesian) [url](https://github.com/oap-project/recdp/blob/master/examples/notebooks/recsys2020/recsys_for_friesian_integration.ipynb)
* DLRM example [url](https://github.com/oap-project/recdp/blob/master/examples/notebooks/dlrm/DLRM_Performance.ipynb)
* DIEN example [url](https://github.com/oap-project/recdp/blob/master/examples/notebooks/dien/dien_data_process.ipynb)

## LICENSE
* Apache 2.0

## Dependency
* Spark 3.x
* python 3.*
