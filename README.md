# Intel® End-to-End AI Optimization Kit for DLRM
## Original source disclose
Source repo: https://github.com/facebookresearch/dlrm

---

# Quick Start

## Install
```
git checkout AIDK_Ray
git submodule update --init --recursive
cd dlrm_all/dlrm
sh patch_dlrm.sh
```

## Environment setup
```
cd ${path_to_e2eaiok}/Dockerfile-ubuntu18.04/
docker build -t e2eaiok-pytorch . -f DockerfilePytorch
```

## Activate docker and conda
```
cd ${path_to_e2eaiok}
docker run --shm-size=300g -it --privileged --network host --device=/dev/dri -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch /bin/bash

source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/
conda activate pytorch_mlperf

```

## Docker environment setup
```
# passwordless config
cd /home/vmagent/app/e2eaiok/scripts
sudo service ssh start
bash config_passwdless_ssh.sh ${other_train_node}

# config node ip
vim /home/vmagent/app/e2eaiok/dlrm_all/dlrm/hosts
    # head node ip
    # worker nodes ip

# Optional: If dataset is loaded from other hdfs during training
vim $HADOOP_HOME/etc/hadoop/core-site.xml
    # set 'fs.defaultFS'
```

## Start Ray cluster
```
# install raydp ray
pip install raydp-nightly

# head node
OMP_NUM_THREADS=** && ray start --head --port 5678 --dashboard-host 0.0.0.0 --object-store-memory 268435456000 --system-config='{"object_spilling_threshold":0.98}'

# worker node
OMP_NUM_THREADS=** && ray start --address='10.112.228.4:5678' --object-store-memory 268435456000
```


## Download Dataset
```
Download the raw data files day_0.gz, ...,day_23.gz from https://labs.criteo.com/2013/12/download-terabyte-click-logs/ and unzip them
```

## RecDP Data Process
```
# check if raw data has been downloaded
ls /home/vmagent/app/dataset/criteo/raw_data
day_0  day_10  day_12  day_14  day_16  day_18  day_2   day_21  day_23  day_4  day_6  day_8
day_1  day_11  day_13  day_15  day_17  day_19  day_20  day_22  day_3   day_5  day_7  day_9

# install recdp
pip install pyrecdp

# spark config
cd /home/vmagent/app/e2eaiok/config/
cp spark-defaults.conf /home/spark-3.2.1-bin-hadoop3.2/conf/
mkdir -p /home/vmagent/app/e2eaiok/spark_data_processing/spark_local_dir
mkdir -p /home/mnt/applicationHistory

# set parameters
/home/vmagent/app/e2eaiok/dlrm_all/dlrm/data_processing/config.yaml

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

## All In One
```
bash run_all_in_one.sh
```