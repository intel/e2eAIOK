# RecDP v2.0

# INTRODUCTION
RecDP v2.0 is aiming to provide auto Data Prepartion upon spark and pandas.
* Auto Feature Enrich including:
    * feature transformation(datetime, geo_info, text_nlp, url, etc.)
    * feature cross(aggregation transformation - sum, avg, count, etc.)
* Auto anomalies detection
* Feature Profiling Visualizer
* ML/DL connector:
    * numpy based - xgboost/lightgbm
    * pytorch tensor based
    * dgl graph
    * pyG graph

![RecDP v2.0 Overview](resources/recdp_20_overview.png)

# Getting Start
## use docker to setup pyrecdp
```
git clone --single-branch --branch RecDP_v2.0 https://github.com/intel-innersource/frameworks.bigdata.AIDK.git
cd frameworks.bigdata.AIDK/RecDP
python3 scripts/start_e2eaiok_docker.py
#python3 scripts/start_e2eaiok_docker.py --proxy "http://ip:port"
# open browser with http://hostname:8888
```

## Quick Example
> NYC Taxi fare 55M records, RecDP took 350secs and featuretools took 2908secs
> Below is RecDP codes and output, for featuretools script, please see [NYC Taxi fare auto data prepration](examples/notebooks/autofe/FeatureWrangler.ipynb)
```
import pandas as pd
train_data = pd.read_csv("nyc_taxi_fare_cleaned.csv")

# use recdp to do auto feature wrangling
from pyrecdp.autofe import FeatureWrangler
pipeline = FeatureWrangler(dataset=train_data, label="reply")

# switch between spark or pandas
transformed_train_data = pipeline.fit_transform(engine_type = 'spark')
```
```
After analysis, we detect and decided to include below steps in pipeline:
"Stage 0: [<class 'pyrecdp.primitives.generators.dataframe.DataframeConvertFeatureGenerator'>]",
"Stage 1: [<class 'pyrecdp.primitives.generators.fillna.FillNaFeatureGenerator'>, <class 'pyrecdp.primitives.generators.type.TypeInferFeatureGenerator'>, <class 'pyrecdp.primitives.generators.geograph.CoordinatesInferFeatureGenerator'>]"
"Stage 2: [<class 'pyrecdp.primitives.generators.datetime.DatetimeFeatureGenerator'>, <class 'pyrecdp.primitives.generators.geograph.GeoFeatureGenerator'>]"
"Stage 3: [<class 'pyrecdp.primitives.generators.dataframe.DataframeTransformFeatureGenerator'>]",
'Stage 4: []',
'Stage 5: []',
'Stage 6: []'
```
```
# output log, spark based
# enriched from 21 features to 41 features
train_data shape is (54315955, 7)
read train data from csv took 45.0155632654205 sec
initiate autofe pipeline took 3.5842366172000766 sec
DataframeConvert partition pandas dataframe to spark RDD took 42.148 secs
DataframeTransform took 292.536 secs, processed 54315955 rows with num_partitions as 200
DataframeTransform combine to one pandas dataframe took 6.072 secs
transform took 348.05814038962126 sec
transformed shape is (54315955, 15)
```

# More Examples
## Auto Feature Enrich Examples
* [NYC Taxi fare auto data prepration](examples/notebooks/autofe/FeatureWrangler.ipynb): An example to show how RecDP_v2.0 automatically generating datetime and geo features upon 55M records. Tested with both Spark and Pandas(featuretools) as compute engine, show 21x speedup by spark.

* [twitter auto data prepration](examples/notebooks/autofe/FeatureWrangler-recsys.ipynb): An example to show how RecDP_v2.0 automatically generating datetime, nlp features upon 14M records. Tested with both Spark and Pandas(featuretools) as compute engine, show 10x speedup by spark.

* amazon products review: To be added in near future

## Data Profiler Examples
* [NYC Taxi fare Profiler](examples/notebooks/autofe/FeatureProfiler.ipynb), [snapshot](resources/FeatureProfiler_NYC.png): An example to show RecDP_v2.0 to profile data, including infer the potential data type, generate data distribution charts.

* [twitter Profiler](examples/notebooks/autofe/FeatureProfiler_recsys.ipynb), [snapshot](resources/FeatureProfiler_recsys.png): An example to show RecDP_v2.0 to profile data, including infer the potential data type, generate data distribution charts.

## Feature Cross

* multiple table feature cross: To be added in near future

* single table feature cross: To be added in near future

## connector example

* To be added in near future

## LICENSE
* Apache 2.0

## Dependency
* Spark 3.x
* python 3.*
