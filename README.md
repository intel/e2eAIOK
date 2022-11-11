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
> Important: default dataset and spark dir is same as code. Please make sure there are ~1000G to run DLRM small scale test. Full Scale need 2.5T.
```
cd ${path_to_e2eaiok}
python3 scripts/start_e2eaiok_docker.py -b pytorch_mlperf

# To configure spark dir / proxy / dataset using below cmdline
# python3 scripts/start_e2eaiok_docker.py -b pytorch_mlperf -b ${local_host_name} --proxy "http://ip:port" --spark_shuffle_dir "" --dataset_path ""
```

###  Enter docker and Activate conda
```
sshpass -p docker ssh ${local_host_name} -p 12346
```

------
## How to run

### Click to lauch end to end AI pipeline
```
conda activate pytorch_mlperf
cd /home/vmagent/app/e2eaiok/dlrm_all/dlrm/; bash run_aiokray_dlrm.sh local_small ${current_node_ip}
```

------
## How to clean ENV
```
docker rm e2eaiok-pytorch-mlperf -f
# remove port as 12346 in known_hosts
```