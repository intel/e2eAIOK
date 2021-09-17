# How to run

```
# download kaggle dataset
python
>> import kaggle
>>         kaggle.api.authenticate()
>>         kaggle.api.dataset_download_files('stackoverflow/stacksample', path=SO_PATH, unzip=True, quiet=False)

# or download from website https://www.kaggle.com/stackoverflow/stacksample
        
# cp stack-overflow folder to ${current_path}/recdp/examples/python_tests/haystack_sod/

# run elastic search server on master node(if you are running on multiple node, choose one)
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2

# run below codes to all physical nodes
# Fix the recdp path to align with your env
docker run -it --net=host -v /mnt/nvme2/chendi/BlueWhale/recdp:/home/vmagent/app/recdp -v /mnt/nvme2/chendi/BlueWhale/applications.ai.appliedml.haystack/:/home/haystack -v /mnt/nvme2/chendi/BlueWhale/applications.ai.appliedml.haystack/haystack/:/home/user/haystack xuechendi/haystack-cpu-recdp /bin/bash

# after enter docker, start hdfs, spark master and workers

# for first time, format hdfs
${HADOOP_HOME}/bin/hdfs namenode -format

# start hdfs
${HADOOP_HOME}/sbin/start-dfs.sh

# start spark
vim ${SPARK_HOME}/conf/workers
# add all workers hostname
# Example
sr602
sr603
sr604
...

# start spark master on one of your node
${SPARK_HOME}/sbin/start-master.sh

# start spark workers on all nodes
${SPARK_HOME}/sbin/start-worker.sh spark://${master_hostname}:7077

# start spark history server
${SPARK_HOME}/sbin/start-history-server.sh

# run gen sod script
cd /home/vmagent/app/recdp/examples/python_tests/haystack_sod/

# doing spark configuration in gen_sod_to_documentstore.py
EMBEDDING_PARALLELISM = 8     # num_nodes * 2
NUMCORES_PER_SOCKET = 64
EMBEDDING_BATCH_SIZE = 262144 # 256k
ELASTIC_SERVER = ${master_hostname}

python gen_sod_to_documentstore.py
```

![image](https://user-images.githubusercontent.com/4355494/133181494-b55a9366-b61b-4cad-a498-907949ec67ca.png)

##### checkout inferencing progress in spark executor stderr log
![image](https://user-images.githubusercontent.com/4355494/133181797-c1a2a1fe-230c-45ff-a536-89ed143a51c8.png)



