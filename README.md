# Distributed training and inference on Ray and Spark with Intel® End-to-End Optimization Kit 

## Overview
Modern recommendation systems require a complex pipeline to handle both data processing and feature engineering at a tremendous scale, while promising high service level agreements for complex deep learning models. Usually, this leads to two separate clusters for data processing and training: a data process cluster, usually CPU based to process huge dataset (terabytes or petabytes) stored in distributed storage system, and a training cluster, usually GPU based for training. This separate data processing and training cluster results a complex software stack, heavy data movement cost.
Meanwhile, Deep learning models were commonly used for recommendation systems, quite often, those models are over-parameterized. It takes up to several days or even weeks to training a heavy model on CPU. 
This workflow trying to address those pain points: unifies both data processing and training on Ray – the open source project that make it simple to scale any compute-intensive python workloads, and then optimize the E2E process, especially the training part, shorten training time weight lighter models, maintaining same accuracy, and delivers high-throughput models for inference with Intel® End-to-End Optimization Kit.

## How it works 
![image](https://github.com/intel-innersource/frameworks.bigdata.AIDK/assets/6396930/fb9ada53-ca84-4158-9562-261b6933dfe0)

## Original source disclose
Source repo: https://github.com/facebookresearch/dlrm

---

# Quick Start

## Download Dataset
```
# Download the raw data files day_0.gz, ...,day_23.gz from https://labs.criteo.com/2013/12/download-terabyte-click-logs/ and unzip them

curl -O https://storage.googleapis.com/criteo-cail-datasets/day_{`seq -s "," 0 23`}.gz
for i in `seq 0 23`;do gzip -d day_${i}.gz;done
```
>Note: Make sure the network connections work well for downloading the datasets.


## Prerequisites
>Important: please make sure below cmdlines are executed in all nodes, including head node and workers nodes
```
export path_to_e2eaiok=`pwd`/frameworks.bigdata.AIDK
git clone https://github.com/intel-innersource/frameworks.bigdata.AIDK.git
cd ${path_to_e2eaiok}
git checkout AIDK_Ray
git submodule update --init --recursive
cd dlrm_all/dlrm
sh patch_dlrm.sh
```

## Docker

### Docker Setup
>Important: please make sure below cmdlines are executed in all nodes, including head node and workers nodes
```
cd ${path_to_e2eaiok}/Dockerfile-ubuntu18.04/
docker build -t e2eaiok-pytorch . -f DockerfilePytorch
```

###  Enter docker and Activate conda
>Important: please make sure below cmdlines are executed in all nodes, including head node and workers nodes
```
cd ${path_to_e2eaiok}
docker run --shm-size=300g -it --privileged --network host --device=/dev/dri -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch /bin/bash
conda activate pytorch_mlperf
sudo service ssh start
pip install raydp-nightly pyrecdp
```
------
### Inside Docker environment setup
>Important: Now only need to execute on head node.
```
# passwordless config
cd /home/vmagent/app/e2eaiok/scripts
bash config_passwdless_ssh.sh ${other_train_node}

# config node ip
vim /home/vmagent/app/e2eaiok/dlrm_all/dlrm/hosts
    # head node ip
    # worker nodes ip

# Optional: If dataset is loaded from other hdfs during training
vim $HADOOP_HOME/etc/hadoop/core-site.xml
    # set 'fs.defaultFS'
```
> Important: 
```
# making sure your HDFS cluster is running with 2T free capacity
# enter ${head_node}:9870 in browser to check

# upload local criteo data to HDFS
${HADOOP_HOME}/bin/hdfs dfs -put /home/vmagent/app/dataset/criteo/raw_data/* hdfs://${hdfs_master_node}:9000/DLRM_DATA/
```

------
## How to run
### config and start service
>Important: Now only need to execute on head node.
```
# set parameters in below two files
vim /home/vmagent/app/e2eaiok/dlrm_all/dlrm/data_processing/config.yaml
vim /home/vmagent/app/e2eaiok/dlrm_all/dlrm/data_processing/config_infer.yaml
    # set 'hdfs_node'
    # set 'output_folder'
    # set 'config.spark.local.dir'

# execute on head node
OMP_NUM_THREADS=** && ray start --head --port 5678 --dashboard-host 0.0.0.0 --object-store-memory 268435456000 --system-config='{"object_spilling_threshold":0.98}'

ssh ${worker_node} OMP_NUM_THREADS=** && ray start --address='10.112.228.4:5678' --object-store-memory 268435456000
```

### Click to lauch end to end AI pipeline
```
bash run_all_in_one.sh
```

------
## Optional: Run step by step

### Data Process
```
# perform data process workflow
cd /home/vmagent/app/e2eaiok/dlrm_all/dlrm
bash run_data_process.sh

# check result
cat data_processing/data_info.txt
```

## Training
```
bash run_train.sh
```

## Inference
```
bash run_inference.sh
```