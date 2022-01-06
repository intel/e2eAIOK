
## Quick start


#### Prepare dataset
```
# We assume dataset path is /root/dataset/tpcxai
# If you use different path, change the path in commands accordingly.
```

#### Git clone hydro.ai repo
```
# We assume hydro.ai repo path is /root/hydro/frameworks.bigdata.bluewhale
# If you use different path, change the path in commands accordingly.
cd /root/hydro
git clone https://github.com/intel-innersource/frameworks.bigdata.bluewhale.git
```

#### Prepare docker
```
# pull hydro docker image and run
docker run -it --privileged --network host --device=/dev/dri -v /root/dataset/tpcxai:/home/vmagent/app/dataset/tpcxai -v /root/hydro/frameworks.bigdata.bluewhale:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ docker.io/xuechendi/oneapi-aikit:hydro.ai /bin/bash

# modify spark conf (change back to your config with different num_instances and num_cores)
export master_hostname=`hostname`
vim /home/spark-3.2.0-bin-hadoop3.2/conf/spark-defaults.conf

spark.eventLog.enabled             true
spark.eventLog.dir                 file:///home/mnt/applicationHistory
spark.history.fs.logDirectory      file:///home/mnt/applicationHistory
spark.network.timeout              7200s
spark.history.ui.port              18081
spark.local.dir                    /home/mnt/spark_local_dir

spark.network.timeout 12000s
spark.executor.heartbeatInterval 10000s
spark.sql.adaptive.enabled true

spark.master                       spark://${master_hostname}:7077
spark.executor.memory              160g
spark.executor.instances           2
spark.executor.cores               20
spark.executor.memoryOverhead      16g
spark.driver.memory                10g
spark.rpc.message.maxSize          1024
spark.executorEnv.NUMBA_CACHE_DIR  /tmp

# start ssh, hdfs and spark service
service ssh start
conda activate tensorflow
source /etc/profile.d/spark-env.sh
export PATH=${PATH}:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin
${HADOOP_HOME}/bin/hdfs namenode -format
${HADOOP_HOME}/sbin/start-dfs.sh
${SPARK_HOME}/sbin/start-master.sh
${SPARK_HOME}/sbin/start-worker.sh spark://${master_hostname}:7077
${SPARK_HOME}/sbin/start-history-server.sh

# copy dataset from local to hdfs
hdfs dfs -mkdir -p /user/root/output/data
hdfs dfs -copyFromLocal /home/vmagent/app/dataset/tpcxai/* /user/root/output/data/

# config network proxy (Optional)
source /home/vmagent/app/hydro.ai/scripts/config_proxy

# conda env (Optional, can be skipped if the packages already installed)
conda activate tensorflow
rm -rf /opt/intel/oneapi/tensorflow/2.5.0/lib/python3.7/site-packages/six-1.16.0.dist-info/
pip install fsspec
pip install tensorflow-addons==0.15.0 --no-deps
pip install typeguard opencv-python
pip install dlib
pip install petastorm --no-deps
pip install pandas diskcache packaging pyzmq
```

#### Run TPCxAI UC09
```
# enable oneapi
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# activate tensorflow conda env
conda activate tensorflow
# set PYSPARK env variables
export PYSPARK_PYTHON=/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python
export PYSPARK_WORKER_PYTHON=/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python
# Use hydro.ai API
SIGOPT_API_TOKEN=${TOKEN} python /home/vmagent/app/hydro.ai/run_hydroai.py --data_path /home/vmagent/app/dataset/tpcxai --model_name tpcxai09 --conf /home/vmagent/app/hydro.ai/conf/TPCxAI/hydroai_defaults_tpcxai_uc9_example.conf
# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python /home/vmagent/app/hydro.ai/SDA/SDA.py --data_path /home/vmagent/app/dataset/tpcxai --model_name tpcxai09 --conf /home/vmagent/app/hydro.ai/conf/TPCxAI/hydroai_defaults_tpcxai_uc9_example.conf
```
