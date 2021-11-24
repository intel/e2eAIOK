# Deep Interest Evolution Network for Click-Through Rate Prediction
https://arxiv.org/abs/1809.03672
original-resource: https://github.com/alibaba/ai-matrix

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

# copy processed data from output to current folder
```
cp ${recdp_project_path}/examples/python_tests/dien/output/ ./
cp ${recdp_project_path}/examples/python_tests/dien/j2c_test/*info ./
```

When you see the files below in output folder.
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info
## train model
```
# remember to change attached folder to where this fold is
./run_docker
cd /home/xxx/dien
./train.sh
```

and to run inference only, use
```
./infer.sh
```
