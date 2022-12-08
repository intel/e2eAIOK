# Distributed training and inference on Ray and Spark with Intel® End-to-End Optimization Kit 

## Overview
Modern recommendation systems require a complex pipeline to handle both data processing and feature engineering at a tremendous scale, while promising high service level agreements for complex deep learning models. Usually, this leads to two separate clusters for data processing and training: a data process cluster, usually CPU based to process huge dataset (terabytes or petabytes) stored in distributed storage system, and a training cluster, usually GPU based for training. This separate data processing and training cluster results a complex software stack, heavy data movement cost.
Meanwhile, Deep learning models were commonly used for recommendation systems, quite often, those models are over-parameterized. It takes up to several days or even weeks to training a heavy model on CPU. 
This workflow trying to address those pain points: unifies both data processing and training on Ray – the open source project that make it simple to scale any compute-intensive python workloads, and then optimize the E2E process, especially the training part, shorten training time weight lighter models, maintaining same accuracy, and delivers high-throughput models for inference with Intel® End-to-End Optimization Kit.

> Important: original source disclose - https://github.com/facebookresearch/dlrm

## How it works 
![image](https://github.com/intel-innersource/frameworks.bigdata.AIDK/assets/6396930/fb9ada53-ca84-4158-9562-261b6933dfe0)

## Get Started

### Prerequisites
```
git clone https://github.com/intel-innersource/frameworks.bigdata.AIDK.git
cd frameworks.bigdata.AIDK
git checkout AIDK_Ray
git submodule update --init --recursive
sh dlrm_all/dlrm/patch_dlrm.sh
```

### Docker
```
# prepare a folder for dataset
cd frameworks.bigdata.AIDK
mkdir -p ../e2eaiok_dataset
cur_path=`pwd`

# run docker
docker run -it --shm-size=300g --privileged --network host --device=/dev/dri -v ${cur_path}/../e2eaiok_dataset/:/home/vmagent/app/dataset -v ${cur_path}:/home/vmagent/app/e2eaiok -v ${cur_path}/../spark_local_dir/:/home/vmagent/app/spark_local_dir -w /home/vmagent/app/ --name e2eaiok-ray-pytorch e2eaiok/e2eaiok-ray-pytorch /bin/bash
```

### How to run (run below cmds inside docker)
```
# active conda
conda activate pytorch_mlperf

# if behind proxy, please set proxy firstly
# export https_proxy=http://{ip}:{port}

# kaggle test
# For kaggle test, ~45G disk space is required
cd /home/vmagent/app/e2eaiok/dlrm_all/dlrm/; bash run_aiokray_dlrm.sh kaggle ${current_node_ip}

```

### Now you have completed the test

------

## Useful Resources

## Recommended Hardware and OS

* recommend to use ubuntu20.04 as Host OS
* memory size is over 250G
* disk capacity requirement
    * For kaggle run, ~45G is required for both spark_shuffle_dir and dataset, 10G for spark shuffle and 35G for dataset
    * For criteo small run, at lease ~300G is required for spark_shuffle_dir and ~500G is required for dataset
    * For criteo full run, at lease ~1500G is required for spark_shuffle_dir(3 nodes, 500G each) and ~1000G is required for dataset on HDFS, another ~2000G is required for dataset on head node local disk.

### Dataset
> Note: For kaggle run, train.csv and test.csv are required.

kaggle: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz

> Note: For criteo small run, day_0, day_1, day_2, day_3, day_23 are required.

> Note: For criteo full test, day_0-day_23 are required

criteo small and criteo full: https://labs.criteo.com/2013/12/download-terabyte-click-logs/

### step by step guide

[option1] Build docker for single or multiple node and enter docker with click-to-run script
```
python3 scripts/start_e2eaiok_docker.py
sshpass -p docker ssh ${local_host_name} -p 12346
# If you met any network/package not found error, please follow log output to do the fixing and re-run above cmdline.

# If you are behind proxy, use below cmd
# python3 scripts/start_e2eaiok_docker.py --proxy "http://ip:port"
# sshpass -p docker ssh ${local_host_name} -p 12346

# If you disk space is limited, you can specify spark_shuffle_dir and dataset_path to different mounted volumn
# python3 scripts/start_e2eaiok_docker.py --spark_shuffle_dir "" --dataset_path ""
# sshpass -p docker ssh ${local_host_name} -p 12346
```

[option2] Build docker manually
```
# prepare a folder for dataset
cd frameworks.bigdata.AIDK
cur_path=`pwd`
mkdir -p ../e2eaiok_dataset

# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -P Dockerfile-ubuntu18.04/ -O Dockerfile-ubuntu18.04/miniconda.sh

# build docker from dockerfile
docker build -t e2eaiok-ray-pytorch Dockerfile-ubuntu18.04 -f Dockerfile-ubuntu18.04/DockerfilePytorch
# if you are behine proxy
# docker build -t e2eaiok-ray-pytorch Dockerfile-ubuntu18.04 -f Dockerfile-ubuntu18.04/DockerfilePytorch --build-arg http_proxy={ip}:{port} --build-arg https_proxy=http://{ip}:{port}

# run docker
docker run -it --shm-size=300g --privileged --network host --device=/dev/dri -v ${cur_path}/../e2eaiok_dataset/:/home/vmagent/app/dataset -v ${cur_path}:/home/vmagent/app/e2eaiok -v ${cur_path}/../spark_local_dir/:/home/vmagent/app/spark_local_dir -w /home/vmagent/app/ --name e2eaiok-ray-pytorch e2eaiok-ray-pytorch /bin/bash
```

### Test with other dataset (run cmd inside docker)
```
# active conda
conda activate pytorch_mlperf

# if behind proxy, please set proxy firstly
# export https_proxy=http://{ip}:{port}

# criteo test
cd /home/vmagent/app/e2eaiok/dlrm_all/dlrm/; bash run_aiokray_dlrm.sh criteo_small ${current_node_ip}
```

### Test full pipeline manually (run cmd inside docker)
```
# active conda
conda activate pytorch_mlperf

# if behind proxy, please set proxy firstly
# export https_proxy=http://{ip}:{port}

# prepare env
bash run_prepare_env.sh ${run_mode} ${current_node_ip}

# data process
bash run_data_process.sh ${run_mode} ${current_node_ip}

# train
bash run_train.sh ${run_mode} ${current_node_ip}

# inference
bash run_inference.sh ${run_mode} ${current_node_ip}
```

## Support

jian.zhang@intel.com

