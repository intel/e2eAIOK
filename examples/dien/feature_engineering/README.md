## prepare data
### use recdp
```
git clone https://github.com/oap-project/recdp.git
cd recdp/examples/python_tests/dien/
# modify num_cores and memory_capacity
vim j2c_spark.py
# fix at line 87 to 93
spark = SparkSession.builder.master('local[${num_core}]')\
        .appName("dien_data_process")\
        .config("spark.driver.memory", "${memory_size}G")\
        .config("spark.executor.cores", "${num_core}")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()

vim dien_data_process.py
#fix line 155 to 161
spark = SparkSession.builder.master(...

# call run
./download_dataset
./run
```
