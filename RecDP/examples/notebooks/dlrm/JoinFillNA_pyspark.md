```python
spark.stop()
```

## DLRM Categorify Performance Analysis

#### 1. FillNA performance
1. Baseline: Read from HDFS and write back, took 12min 35s
[Baseline](http://sr602:18080/history/application_1615165886357_0496/SQL/execution/?id=53) [code block](#FillNA-overhead-evaluation)

2. Read from HDFS and do fillNA to all columns and write, took 11min 57s 
[FillNAToAll](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=107) [code block](#Step3:-Fill-NA)

3. Read from HDFS do fillNA to c1 column and write, took  12min 52s
[FillNAToOne](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=215) [code block](#FillNA-overhead-evaluation)

4. Read from HDFS do fillNA to c14-c39 column and write, took 12min 32s
[FillNATo26](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=216) [code block](#FillNA-overhead-evaluation)


##### Conclusion
* FillNA project has very small performance overhead scaling from one column to 39 columns

#### 2. Categorify & FillNA performance
1. Separate Join and FillNA to two phases, took 31min 49s + 11min 57s
[step 1: Categorify](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=105)
[step 2: FillNA](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=107)

2. AllInOne as one phase, took 34min 24s
[allInOne](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=214)

##### Conclusion
* FillNA project inside WSCG brought overhead, should avoid that!!!!
* Reason for that is due to w/ project BHJ WSCG need no memcpy, comparison as below
* firstly we do fillNA in the same WSCG with categorify, which took 54min [Link](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=161)
* Since project should not be that heavy, we seperate categorify and FillNA to two stage, and it only took 34 min(24 + 10)
[Link](http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=214)

### Script


```python
###### Start spark job ######

from DataProcessUtils.init_spark import *
from DataProcessUtils.utils import *
from RecsysSchema import RecsysSchema

import logging
from timeit import default_timer as timer
import os
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd

path_prefix = "hdfs://"
csv_folder = "/dlrm/csv_raw_data/"
parquet_folder = "/dlrm/parquet_raw_data/"
file = "/dlrm/raw_data/day_0"
#path = [os.path.join(path_prefix, folder, file) for file in files]
csv_path = os.path.join(path_prefix, csv_folder)
parquet_path = os.path.join(path_prefix, parquet_folder)
#path = os.path.join(path_prefix, file)


##### 1. Start spark and initialize data processor #####
t0 = timer()
spark = SparkSession\
    .builder\
    .master('yarn')\
    .appName("DLRM_JOIN_FILLNA") \
    .getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
```

### Convert CSV to parquet


```python
#LABEL_COL = 0
#INT_COLS = list(range(1, 14))
#CAT_COLS = list(range(14, 40))
#label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
#int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
#str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]
#schema = StructType(label_fields + int_fields + str_fields)

#files = ["day_%d" % i for i in range(0, 24)]
#for file_name in files:
#    df = spark.read.schema(schema).option('sep', '\t').csv(os.path.join(path_prefix, csv_folder, file_name))
#    df.write.format('parquet').mode('overwrite').save(os.path.join(path_prefix, parquet_folder, file_name))
```

### Generate Dictionary for all Category columns
* DLRM categorify applied to 26 columns, _c14 -> _c39, corresponding dictionary length and size listed as below

| col_id | numRows  | broadcastsize | broadcast elapse time |
| ------ | -------  | ----------- | ----------- |
| _c14   | 7912888	| 108.4 MiB	  | 7 s   |
| _c15   | 33822	| 35.8 MiB	  | 2 s   |
| _c16   | 17138	| 2.3 KiB	  | 92 ms |
| _c17   | 7338	    | 3.8 KiB	  | 0.1 s |
| _c18   | 20045	| 4.2 MiB	  | 0.9 s |
| _c19   | 3	    | 95.1 MiB	  | 6 s   |
| _c20   | 7104	    | 191.0 KiB	  | 0.3 s |
| _c21   | 1381	    | 978.0 B	  | 0.4 s |
| _c22   | 62	    | 19.6 KiB	  | 0.6 s |
| _c23   | 5554113	| 4.1 KiB	  | 88 ms |
| _c24   | 582468	| 683.0 B	  | 0.6 s |
| _c25   | 245827	| 7.9 MiB	  | 0.9 s |
| _c26   | 10	    | 39.8 KiB	  | 0.6 s |
| _c27   | 2208	    | 73.9 MiB	  | 5 s   |
| _c28   | 10666	| 169.1 KiB	  | 0.6 s |
| _c29   | 103	    | 222.0 B	  | 0.5 s |
| _c30   | 3	    | 3.4 MiB	  | 0.9 s |
| _c31   | 967	    | 3.7 KiB	  | 0.6 s |
| _c32   | 14	    | 26.9 KiB	  | 0.5 s |
| _c33   | 8165895	| 116.8 KiB	  | 0.6 s |
| _c34   | 2675939	| 297.0 KiB	  | 0.6 s |
| _c35   | 7156452	| 221.0 B	  | 0.5 s |
| _c36   | 302515	| 115.3 KiB	  | 0.1 s |
| _c37   | 12021	| 268.1 KiB	  | 0.6 s |
| _c38   | 96	    | 105.1 MiB	  | 7 s   |
| _c39   | 34	    | 503.2 KiB	  | 0.7 s |


```python
#files = [os.path.join(parquet_folder, "day_%d" % i) for i in range(0, 24)]
#df = spark.read.parquet(*files)
```


```python
#def gen_dict_df(input_df, i):
#    singular_df = input_df.select(i)
#    singular_df = singular_df.filter("%s is not null" % i)
#    singular_df = singular_df.groupBy(i).count()
#    singular_df = singular_df.withColumnRenamed('count', 'model_count').withColumnRenamed(i, 'data')
#    return singular_df
```


```python
#cols = ['_c%d' % i for i in CAT_COLS]
#for col in cols:
#    col_df = gen_dict_df(df, col).filter("model_count >= 15")  # frequency_limit
#    col_df.cache()
#    col_df.write.format('parquet').mode(
#                'overwrite').save("/dlrm/models/%s/" % col)
#    print("%s saved, total numRows is %d" % (col, col_df.count()))
```

### Categorify for all category cols, start from here!


```python
import pyspark.sql.functions as f
def get_mapping_udf(broadcast_data, default=None):
    first_value = next(iter(broadcast_data.values()))
    if not isinstance(first_value, int) and not isinstance(first_value, str) and not isinstance(first_value, float):
        raise NotImplementedError

    broadcast_dict = spark.sparkContext.broadcast(
        broadcast_data)

    def get_mapped(x):
        broadcast_data = broadcast_dict.value
        if x in broadcast_data:
            return broadcast_data[x]
        else:
            return default

    # switch return type
    if isinstance(first_value, int):
        return udf(get_mapped, IntegerType())
    if isinstance(first_value, str):
        return udf(get_mapped, StringType())
    if isinstance(first_value, float):
        return udf(get_mapped, FloatType())
    
def categorify_with_udf(df, dict_df, i):
    sorted_data = dict_df.orderBy('model_count').collect()
    dict_data = dict((row['data'], idx) for (row, idx) in zip(
        sorted_data, range(0, len(sorted_data))))
    udf_impl = get_mapping_udf(dict_data)
    df = df.withColumn(i, udf_impl(f.col(i)))
    return df

def categorify_with_join(df, dict_df, i):
    dict_df = dict_df.withColumn('id', row_number().over(
        Window.orderBy(desc('model_count')))).withColumn('id', f.col('id') - 1).select('data', 'id')
    df = df.join(dict_df.hint('shuffle_hash'), f.col(i) == dict_df.data, 'left')\
           .withColumn(i, dict_df.id).drop("id", "data")
    return df

def categorify_with_bhj(df, dict_df, i):
    dict_df = dict_df.withColumn('id', row_number().over(
        Window.orderBy(desc('model_count')))).withColumn('id', f.col('id') - 1).select('data', 'id')
    df = df.join(dict_df.hint('broadcast'), f.col(i) == dict_df.data, 'left')\
           .withColumn(i, dict_df.id).drop("id", "data")
    return df

def categorify_strategy_decision_maker(dict_dfs):
    small_cols = []
    long_cols = []
    huge_cols = []
    for (col, dict_df) in dict_dfs:
        if (dict_df.count() > 1000000000):
            huge_cols.append(col)
        elif (dict_df.count() > 1000000000):
            long_cols.append(col)
        else:
            small_cols.append(col)
    return {'short_dict': small_cols, 'long_dict': long_cols, 'huge_dict': huge_cols}

def categorify(df, dict_dfs):
    strategy = categorify_strategy_decision_maker(dict_dfs)
    # for short dict, we will do bhj
    for (col, dict_df) in dict_dfs:
        if col in strategy['short_dict']:
            df = categorify_with_bhj(df, dict_df, col)    
    # for long dict, we will do shj all along    
    for (col, dict_df) in dict_dfs:
        if col in strategy['long_dict']:
            df = categorify_with_bhj(df, dict_df, col)
    # for huge dict, we will do shj seperately
    for (col, dict_df) in dict_dfs:
        if col in strategy['huge_dict']:
            df = categorify_with_join(df, dict_df, col)
            tmp_path = "/dlrm/tmp/%s/" % uuid.uuid1()
            df.write.format('parquet').mode('overwrite').save(tmp_path)
            df = spark.read.parquet(tmp_path)
    return df

```

### Step 1: Load all pre-generated dicts and original data


```python
import uuid

files = [os.path.join(parquet_folder, "day_%d" % i) for i in range(0, 24)]
df = spark.read.parquet(*files)
LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
cols = ['_c%d' % i for i in CAT_COLS]
models_folder = "/dlrm/models/"
dict_dfs = [(col, spark.read.parquet(os.path.join(models_folder, col))) for col in cols]
```

### Step 2 Categorify and FillNA - All In One
* We tried to do short dict / long dict and FillNA seperately, 
* noticed this data is not very memory intensive, so decided to do All In One

##### Notice ! Now we verified option 1 performed better

###### option 1: seperate categorify and fillNA in two stages ######


```python
df = categorify(df, dict_dfs)
cols = ['_c%d' % i for i in INT_COLS + CAT_COLS]
df= df.repartition(2000)
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# save to: /dlrm/tmp/d684684e-a970-11eb-b161-a4bf0121f496
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=162
```

    {'short_dict': ['_c14', '_c15', '_c16', '_c17', '_c18', '_c19', '_c20', '_c21', '_c22', '_c23', '_c24', '_c25', '_c26', '_c27', '_c28', '_c29', '_c30', '_c31', '_c32', '_c33', '_c34', '_c35', '_c36', '_c37', '_c38', '_c39'], 'long_dict': [], 'huge_dict': []}
    CPU times: user 282 ms, sys: 180 ms, total: 462 ms
    Wall time: 34min 24s


###### option 2: categorify and fillNA in same WSCG stages ######


```python
df = categorify(df, dict_dfs)
cols = ['_c%d' % i for i in INT_COLS + CAT_COLS]
print("FillNA to below cols:")
print(cols)
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# save to: /dlrm/tmp/e2b4b890-a965-11eb-b161-a4bf0121f496
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=161
```

    {'short_dict': ['_c14', '_c15', '_c16', '_c17', '_c18', '_c19', '_c20', '_c21', '_c22', '_c23', '_c24', '_c25', '_c26', '_c27', '_c28', '_c29', '_c30', '_c31', '_c32', '_c33', '_c34', '_c35', '_c36', '_c37', '_c38', '_c39'], 'long_dict': [], 'huge_dict': []}
    FillNA to below cols:
    ['_c1', '_c2', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13', '_c14', '_c15', '_c16', '_c17', '_c18', '_c19', '_c20', '_c21', '_c22', '_c23', '_c24', '_c25', '_c26', '_c27', '_c28', '_c29', '_c30', '_c31', '_c32', '_c33', '_c34', '_c35', '_c36', '_c37', '_c38', '_c39']
    CPU times: user 463 ms, sys: 302 ms, total: 765 ms
    Wall time: 54min 24s



```python
###### option 3: using stringIndexer for categorify and fillNA in two stages ######
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def categorify(df, dict_dfs):
    strategy = categorify_strategy_decision_maker(dict_dfs)
    indexers = []
    # for short dict, we will do bhj
    for (col, dict_df) in dict_dfs:
        if col in strategy['short_dict'] or col in strategy['long_dict']:
            indexer = StringIndexer(inputCol=col, outputCol=col+"_index").fit(df)
            indexers.append(indexer)
    # for huge dict, we will do shj seperately
    for (col, dict_df) in dict_dfs:
        if col in strategy['huge_dict']:
            df = categorify_with_join(df, dict_df, col)
            tmp_path = "/dlrm/tmp/%s/" % uuid.uuid1()
            df.write.format('parquet').mode('overwrite').save(tmp_path)
            df = spark.read.parquet(tmp_path)
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    return df

df = categorify(df, dict_dfs)
cols = ['_c%d' % i for i in INT_COLS + CAT_COLS]
#df= df.repartition(2000)
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# save to: /dlrm/tmp/d684684e-a970-11eb-b161-a4bf0121f496
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=162
```

## Below is why we decided to perform 'All In One'
### Step 1: Categorify
#### option 1: use python UDF to do categorify, took 1h 26min 36s  -> Bad
#### option 2: use Join to do categorify, took 28min and 18 min (tried bhj to all 26 dict, took 31min 49s) -> Looks much better
* First round do BHJ to small dict, took 28min
* Second round do BHJ to long dict, took 18 min
* Then tried to do above BHJ at once, took 31min 49s

#### option 3: use scala UDF to do categorify, still wip -> similiar to StringIndexer
#### option 4: use Join to do categorify(BHJ + SHJ) -> very stupid idea
* SHJ will need to do repartition by join key, materialize every round
* SHJ should only apply to huge dict (which BHJ will definitely go OOM case)

### Step 2: FillNA
* Took about 11min 57s

### Step 2: Do categorify for all short disk columns (Categorify part I)


```python
%time categorify_strategy = categorify_strategy_decision_maker(dict_dfs)
print(categorify_strategy)
```

    CPU times: user 14.7 ms, sys: 6.78 ms, total: 21.5 ms
    Wall time: 20.6 s
    {'short_dict': ['_c15', '_c16', '_c17', '_c18', '_c19', '_c20', '_c21', '_c22', '_c24', '_c25', '_c26', '_c27', '_c28', '_c29', '_c30', '_c31', '_c32', '_c36', '_c37', '_c38', '_c39'], 'long_dict': ['_c14', '_c23', '_c33', '_c34', '_c35'], 'huge_dict': []}



```python
###### Option 1: Do Python UDF to all short dist columns ######
###### Comment: Python UDF performed very bad!!!!! ######

def categorify(df, dict_dfs):
    for (col, dict_df) in dict_dfs:
        df = categorify_with_udf(df, dict_df, col)
    return df

cols = categorify_strategy['short_dict']
models_folder = "/dlrm/models/"
dict_dfs = [(col, spark.read.parquet(os.path.join(models_folder, col))) for col in cols]
df = categorify(df, dict_dfs)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# spark plan: http://sr602:18080/history/application_1615165886357_0493/SQL/execution/?id=70
```

    CPU times: user 766 ms, sys: 288 ms, total: 1.05 s
    Wall time: 1h 26min 36s



```python
###### Option 2: part 1 Do BHJ to all short dist columns ######
###### Comment: Since dict are short, BHJ did pretty well ! ######

cols = categorify_strategy['short_dict']
models_folder = "/dlrm/models/"
dict_dfs = [(col, spark.read.parquet(os.path.join(models_folder, col))) for col in cols]
df = categorify(df, dict_dfs)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# spark plan: http://sr602:18080/history/application_1615165886357_0493/SQL/execution/?id=71
```

    CPU times: user 164 ms, sys: 78.4 ms, total: 243 ms
    Wall time: 28min 36s



```python
###### Option 2: part 2 try to do bhj to long dict ######
###### Comment: Since shj need to do repartition based on join key,  ######
######          and long dict seems fit memory well, use BHJ as well ######

def categorify(df, dict_dfs):
    for (col, dict_df) in dict_dfs:
        df = categorify_with_bhj(df, dict_df, col)
    return df
df = spark.read.parquet("/dlrm/tmp/7538e986-a8f6-11eb-914a-a4bf0121f496")
cols = categorify_strategy['long_dict']
models_folder = "/dlrm/models/"
dict_dfs = [(col, spark.read.parquet(os.path.join(models_folder, col))) for col in cols]
df = categorify(df, dict_dfs)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# saved to /dlrm/tmp/7ab036be-a944-11eb-8f99-a4bf0121f496
# spark plan: http://sr602:18080/history/application_1615165886357_0496/SQL/execution/?id=52
```

    CPU times: user 134 ms, sys: 119 ms, total: 253 ms
    Wall time: 18min



```python
###### option 2 (part 1 & part 2) ######
df = categorify(df, dict_dfs)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# spark plan: http://sr602:8088/proxy/application_1615165886357_0497/SQL/execution/?id=105
```

    {'short_dict': ['_c14', '_c15', '_c16', '_c17', '_c18', '_c19', '_c20', '_c21', '_c22', '_c23', '_c24', '_c25', '_c26', '_c27', '_c28', '_c29', '_c30', '_c31', '_c32', '_c33', '_c34', '_c35', '_c36', '_c37', '_c38', '_c39'], 'long_dict': [], 'huge_dict': []}
    CPU times: user 240 ms, sys: 191 ms, total: 432 ms
    Wall time: 31min 49s



```python
###### Option 3: Can we do with scala UDFs ######
###### Comment:  This should be a similiar approach as StringIndexer did ######


```

### Step3: Fill NA


```python
#read, fillNAToAllColumns and write
df = spark.read.parquet("/dlrm/tmp/7ab036be-a944-11eb-8f99-a4bf0121f496")
cols = ['_c%d' % i for i in INT_COLS + CAT_COLS]
#print(cols)
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=107
```

    CPU times: user 108 ms, sys: 73.9 ms, total: 182 ms
    Wall time: 11min 57s


### FillNA overhead evaluation


```python
# baseline: simply read and write
df = spark.read.parquet("/dlrm/tmp/7ab036be-a944-11eb-8f99-a4bf0121f496")
cols = ['_c%d' % i for i in INT_COLS + CAT_COLS]
#print(cols)
df = df.select(*cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=108
```

    CPU times: user 108 ms, sys: 83.9 ms, total: 192 ms
    Wall time: 12min 35s



```python
# fillNA to one column
df = spark.read.parquet("/dlrm/tmp/7ab036be-a944-11eb-8f99-a4bf0121f496")
cols = ['_c%d' % i for i in [1]]
#print(cols)
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=215
```

    CPU times: user 122 ms, sys: 69 ms, total: 191 ms
    Wall time: 12min 52s



```python
# fillNA to all categorify columns
df = spark.read.parquet("/dlrm/tmp/7ab036be-a944-11eb-8f99-a4bf0121f496")
cols = ['_c%d' % i for i in CAT_COLS]
#print(cols)
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# save to: 
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=215
```

    CPU times: user 117 ms, sys: 68.7 ms, total: 185 ms
    Wall time: 12min 32s


### Redundant, skip belows


```python
#def assign_id_with_window(df):
#    windowed = Window.partitionBy('column_id').orderBy(desc('count'))
#    return (df
#            .withColumn('id', row_number().over(windowed))
#            .withColumnRenamed('count', 'model_count'))
#
#def get_column_counts_with_frequency_limit(df, frequency_limit = None):
#    cols = ['_c%d' % i for i in CAT_COLS]
#    df = (df
#        .select(posexplode(array(*cols)))
#        .withColumnRenamed('pos', 'column_id')
#        .withColumnRenamed('col', 'data')
#        .filter('data is not null')
#        .groupBy('column_id', 'data')
#        .count())
#    if frequency_limit:
#        exclude = []
#        default_limit = frequency_limit
#        if default_limit:
#            remain = [x - CAT_COLS[0] for x in CAT_COLS if x not in exclude]
#            df = df.filter((~col('column_id').isin(remain)) | (col('count') >= default_limit))
#    return df
#
#def get_column_models(combined_model):
#    for i in CAT_COLS:
#        model = (combined_model
#            .filter('column_id == %d' % (i - CAT_COLS[0]))
#            .drop('column_id'))
#        yield i, model
#col_counts = get_column_counts_with_frequency_limit(df, 15)
#combined_model_df = assign_id_with_window(col_counts)
#column_models_df_dict = get_column_models(combined_model_df)
#for i, column_models_df in column_models_df_dict:
#    column_models_df.show()
```


```python
cols = ['_c%d' % i for i in [14, 15, 16]]
new_df = df.select(posexplode(array(*cols)))
df.select(*cols).show()
new_df.show()
```

    +--------+--------+--------+
    |    _c14|    _c15|    _c16|
    +--------+--------+--------+
    |62770d79|e21f5d58|afea442f|
    |e5f3fd8d|a0aaffa6|6faa15d5|
    |62770d79|ad984203|62bec60d|
    |    null|710103fd|c73d2eb5|
    |02e197c5|c2ced437|a2427619|
    |8a2b1e43|3fa554c6|0b8e4616|
    |a80f39e1|79782afd|ddb2d2e1|
    |a46c5216|ddc72fb0|49c7ebf8|
    |9318c40b|53c06361|fea787e5|
    |62770d79|ad984203|ddd956c1|
    |ad98e872|3dbb483e|6faa15d5|
    |09fd7fc8|d92abd7e|2826bc68|
    |359aaecc|f501e8d6|43504368|
    |7ffd46c3|710103fd|a1407382|
    |9a8cb066|7a06385f|417e6103|
    |77519961|a0d14bda|b69d8ea9|
    |447072d2|831049b5|02c63370|
    |3a20c9b6|c2ae8fa1|6faa15d5|
    |e5f3fd8d|a15d1051|72181f31|
    |072027fa|21789080|3223c131|
    +--------+--------+--------+
    only showing top 20 rows
    
    +---+--------+
    |pos|     col|
    +---+--------+
    |  0|62770d79|
    |  1|e21f5d58|
    |  2|afea442f|
    |  0|e5f3fd8d|
    |  1|a0aaffa6|
    |  2|6faa15d5|
    |  0|62770d79|
    |  1|ad984203|
    |  2|62bec60d|
    |  0|    null|
    |  1|710103fd|
    |  2|c73d2eb5|
    |  0|02e197c5|
    |  1|c2ced437|
    |  2|a2427619|
    |  0|8a2b1e43|
    |  1|3fa554c6|
    |  2|0b8e4616|
    |  0|a80f39e1|
    |  1|79782afd|
    +---+--------+
    only showing top 20 rows
    



```python
###### option 1: seperate categorify and fillNA in two stages ######
df = categorify(df, dict_dfs)
cols = ['_c%d' % i for i in INT_COLS + CAT_COLS]
df = df.fillna(0, cols)
%time df.write.format('parquet').mode('overwrite').save("/dlrm/tmp/%s/" % uuid.uuid1())
# save to: /dlrm/tmp/d684684e-a970-11eb-b161-a4bf0121f496
# spark plan: http://sr602:18080/history/application_1615165886357_0497/SQL/execution/?id=162
```


    ---------------------------------------------------------------------------

    Py4JJavaError                             Traceback (most recent call last)

    <timed eval> in <module>


    /hadoop/spark/python/pyspark/sql/readwriter.py in save(self, path, format, mode, partitionBy, **options)
        825             self._jwrite.save()
        826         else:
    --> 827             self._jwrite.save(path)
        828 
        829     @since(1.4)


    /hadoop/spark/python/lib/py4j-0.10.9-src.zip/py4j/java_gateway.py in __call__(self, *args)
       1303         answer = self.gateway_client.send_command(command)
       1304         return_value = get_return_value(
    -> 1305             answer, self.gateway_client, self.target_id, self.name)
       1306 
       1307         for temp_arg in temp_args:


    /hadoop/spark/python/pyspark/sql/utils.py in deco(*a, **kw)
        129     def deco(*a, **kw):
        130         try:
    --> 131             return f(*a, **kw)
        132         except py4j.protocol.Py4JJavaError as e:
        133             converted = convert_exception(e.java_exception)


    /hadoop/spark/python/lib/py4j-0.10.9-src.zip/py4j/protocol.py in get_return_value(answer, gateway_client, target_id, name)
        326                 raise Py4JJavaError(
        327                     "An error occurred while calling {0}{1}{2}.\n".
    --> 328                     format(target_id, ".", name), value)
        329             else:
        330                 raise Py4JError(


    Py4JJavaError: An error occurred while calling o1648.save.
    : org.apache.spark.SparkException: Job aborted.
    	at org.apache.spark.sql.execution.datasources.FileFormatWriter$.write(FileFormatWriter.scala:226)
    	at org.apache.spark.sql.execution.datasources.InsertIntoHadoopFsRelationCommand.run(InsertIntoHadoopFsRelationCommand.scala:178)
    	at org.apache.spark.sql.execution.command.DataWritingCommandExec.sideEffectResult$lzycompute(commands.scala:108)
    	at org.apache.spark.sql.execution.command.DataWritingCommandExec.sideEffectResult(commands.scala:106)
    	at org.apache.spark.sql.execution.command.DataWritingCommandExec.doExecute(commands.scala:131)
    	at org.apache.spark.sql.execution.SparkPlan.$anonfun$execute$1(SparkPlan.scala:175)
    	at org.apache.spark.sql.execution.SparkPlan.$anonfun$executeQuery$1(SparkPlan.scala:213)
    	at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:151)
    	at org.apache.spark.sql.execution.SparkPlan.executeQuery(SparkPlan.scala:210)
    	at org.apache.spark.sql.execution.SparkPlan.execute(SparkPlan.scala:171)
    	at org.apache.spark.sql.execution.QueryExecution.toRdd$lzycompute(QueryExecution.scala:122)
    	at org.apache.spark.sql.execution.QueryExecution.toRdd(QueryExecution.scala:121)
    	at org.apache.spark.sql.DataFrameWriter.$anonfun$runCommand$1(DataFrameWriter.scala:944)
    	at org.apache.spark.sql.execution.SQLExecution$.$anonfun$withNewExecutionId$5(SQLExecution.scala:100)
    	at org.apache.spark.sql.execution.SQLExecution$.withSQLConfPropagated(SQLExecution.scala:160)
    	at org.apache.spark.sql.execution.SQLExecution$.$anonfun$withNewExecutionId$1(SQLExecution.scala:87)
    	at org.apache.spark.sql.SparkSession.withActive(SparkSession.scala:763)
    	at org.apache.spark.sql.execution.SQLExecution$.withNewExecutionId(SQLExecution.scala:64)
    	at org.apache.spark.sql.DataFrameWriter.runCommand(DataFrameWriter.scala:944)
    	at org.apache.spark.sql.DataFrameWriter.saveToV1Source(DataFrameWriter.scala:396)
    	at org.apache.spark.sql.DataFrameWriter.save(DataFrameWriter.scala:380)
    	at org.apache.spark.sql.DataFrameWriter.save(DataFrameWriter.scala:269)
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:498)
    	at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
    	at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:357)
    	at py4j.Gateway.invoke(Gateway.java:282)
    	at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
    	at py4j.commands.CallCommand.execute(CallCommand.java:79)
    	at py4j.GatewayConnection.run(GatewayConnection.java:238)
    	at java.lang.Thread.run(Thread.java:748)
    Caused by: org.apache.spark.SparkException: Job 366 cancelled because killed via the Web UI
    	at org.apache.spark.scheduler.DAGScheduler.failJobAndIndependentStages(DAGScheduler.scala:2023)
    	at org.apache.spark.scheduler.DAGScheduler.handleJobCancellation(DAGScheduler.scala:1919)
    	at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleStageCancellation$1(DAGScheduler.scala:1908)
    	at scala.runtime.java8.JFunction1$mcVI$sp.apply(JFunction1$mcVI$sp.java:23)
    	at scala.collection.IndexedSeqOptimized.foreach(IndexedSeqOptimized.scala:36)
    	at scala.collection.IndexedSeqOptimized.foreach$(IndexedSeqOptimized.scala:33)
    	at scala.collection.mutable.ArrayOps$ofInt.foreach(ArrayOps.scala:246)
    	at org.apache.spark.scheduler.DAGScheduler.handleStageCancellation(DAGScheduler.scala:1901)
    	at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:2166)
    	at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:2152)
    	at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:2141)
    	at org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:49)


