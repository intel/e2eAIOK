# Intel® End-to-End AI Optimization Kit for DLRM
## Original source disclose
Source repo: https://github.com/facebookresearch/dlrm

---

# Quick Start

## Install
```
git clone https://github.com/intel/e2eAIOK.git
git submodule update --init --recursive
cd ${path_to_e2eaiok}/modelzoo/dlrm
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
docker run --shm-size=100g -it --privileged --network host --device=/dev/dri -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch /bin/bash

source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/
conda activate pytorch_mlperf

cd /home/vmagent/app/e2eaiok/
python setup.py install

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

# enter data process folder
cd /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/

# install recdp and pyarrow
pip install pyrecdp pyarrow

# start spark service
cp spark-defaults.conf /home/spark-3.2.1-bin-hadoop3.2/conf/
mkdir -p /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/spark_local_dir
mkdir -p /home/mnt/applicationHistory
sh ./start_spark_service.sh

# process data
# IMPORTANT!! dlrm data processing requires huge space
# make sure ~1.4T for /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/
python preprocessing.py
# convert processed data to binary 
python splitconversion.py

# clean up tmp
rm /home/vmagent/app/dataset/criteo/output/tmp

# check result
(pytorch_mlperf) root@sr414:/home/vmagent/app/e2eaiok/modelzoo/dlrm# ll /home/vmagent/app/dataset/criteo/output/
total 683357412
         4096  ./
         4096  ../
          474  day_fea_count.npz
         4096  dicts/
       249856  dlrm_categorified/
        45056  dlrm_categorified_test/
        45056  dlrm_categorified_valid/
        73728  dlrm_parquet_23/
        40960  dlrm_parquet_test/
      1806336  dlrm_parquet_train/
        40960  dlrm_parquet_valid/
  14261971040  test_data.bin
 671231630720  train_data.bin
  14261970880  valid_data.bin
```

## Training
```
bash run_and_time_launch.sh
```

## Inference
```
bash inference.sh
```

## All In One
```
bash run_all_in_one.sh
```