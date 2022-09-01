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
WIP to add
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