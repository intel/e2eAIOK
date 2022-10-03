# Intel® End-to-End AI Optimization Kit - optimized DIEN Workflow
## Original source disclose
Source repo: https://github.com/alibaba/ai-matrix

---

# Prepare Environriment

## set path
```
export path_to_e2eaiok_dataset=`pwd`/e2eaiok_dataset
export path_to_e2eaiok=`pwd`/e2eAIOK
mkdir -p ${path_to_e2eaiok_dataset}
```

## Install Intel® End-to-End AI Optimization Kit for DIEN
```
git clone https://github.com/intel/e2eAIOK.git
git submodule update --init --recursive
cd ${path_to_e2eaiok}/modelzoo/dien/train
sh patch_dien.sh
```

## Build Docker
```
cd ${path_to_e2eaiok}/Dockerfile-ubuntu18.04/
# case 1: No Proxy required
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow

# case 2: proxy is required
vim ~/.docker/config.json
{
        ...
        "proxies": {
                "default": {
                        "httpProxy": "http://${proxy_host}:${proxy_port}",
                        "httpsProxy": "http://${proxy_host}:${proxy_port}",
                        "noProxy": "localhost,::1,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
                }
        }
}
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy --build-arg https_proxy
```

## Download Dataset (if you would like direct train with processed data, skip this step)
```
cd ${path_to_e2eaiok}/modelzoo/dien/feature_engineering/
./download_dataset ${path_to_e2eaiok_dataset}
ls ${path_to_e2eaiok_dataset}/amazon_reviews
j2c_test  output  raw_data
```

## Run docker and activate conda
```
cd ${path_to_e2eaiok}
docker run --shm-size=10g -it --privileged --network host --device=/dev/dri -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-tensorflow /bin/bash
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
cd /home/vmagent/app/e2eaiok/
python setup.py install
apt install numactl
```

---
# Evaluate DIEN solution

## Step By Step guide

### Data Process
```
# Now you are running inside docker, /home/vmagent/app/
pip install pyrecdp
cd /home/vmagent/app/e2eaiok/conf/spark/
cp spark-defaults.conf /home/spark-3.2.1-bin-hadoop3.2/conf/
mkdir -p /home/vmagent/app/e2eaiok/spark_data_processing/spark_local_dir
mkdir -p /home/mnt/applicationHistory
sh start_spark_service.sh 
cd /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/
python preprocessing.py --train
python preprocessing.py --test
```

### Training
```
conda activate tensorflow
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=4
export HOROVOD_CPU_OPERATIONS=CCL
# export MKLDNN_VERBOSE=2
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY="0,32"
export HOROVOD_THREAD_AFFINITY="1,33"
#export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="0,1,32,33"

# test our best parameter
python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien --no_sigopt
```

### distributed training
```
# prepare data
sh /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/split_for_distribute.sh /home/vmagent/app/dataset/amazon_reviews/train ${num_copy}
# copy all slices to distributed nodes as below
# ex: for slice00, do as below and do the same to other nodes
scp /home/vmagent/app/dataset/amazon_reviews/train/local_train_splitByUser.slice00 ${node}://home/vmagent/app/dataset/amazon_reviews/train/local_train_splitByUser
scp /home/vmagent/app/dataset/amazon_reviews/*pkl ${node}://home/vmagent/app/dataset/amazon_reviews/
scp -r /home/vmagent/app/dataset/amazon_reviews/valid/ ${node}://home/vmagent/app/dataset/amazon_reviews/
scp /home/vmagent/app/dataset/amazon_reviews/meta.yaml ${node}://home/vmagent/app/dataset/amazon_reviews/
# done to slice00, repeat for slice01, 02, 03

rm /home/vmagent/app/dataset/amazon_reviews/train/local_train_splitByUser.slice0*
```

```
# prepare nodes env configuration
# IMPORTANT! make sure source and export also performed on all nodes
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=4
export HOROVOD_CPU_OPERATIONS=CCL
# export MKLDNN_VERBOSE=2
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY="0,32"
export HOROVOD_THREAD_AFFINITY="1,33"
#export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="0,1,32,33"
```

```
# ready to run
# make sure all nodes and correct nic listed in conf
vim conf/e2eaiok_defaults_dien_example.conf
ppn: ${num_copy}
iface: ${nic}
hosts:
- ${node1}
- ${node2}
- ${node3}
- ${node4}

# run distributed training
python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien --no_sigopt --conf conf/e2eaiok_defaults_dien_example.conf
```

### use SDA to search best parameter
```
# single node
SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien

# distributed
SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien --conf conf/e2eaiok_defaults_dien_example.conf
```

## Run All in one script
```
sh /home/vmagent/app/e2eaiok/run_dien_all_in_one.sh
```

## Result
```
# after training, you'll see info as below
<!-- 
We found the best model! Here is the model explaination

===============================================
***    Best Trained Model    ***
===============================================
  Model Type: dien
  Model Saved Path: ${path to your result}
  Sigopt Experiment id is None
  === Result Metrics ===
    AUC: 0.8205973396674585
    training_time: 410.986137151718
=============================================== 
-->
```

## Inference
```
cd /home/vmagent/app/e2eaiok/modelzoo/dien/
rm /home/vmagent/app/e2eaiok/modelzoo/dien/train/ai-matrix/dnn_best_model_trained/ -rf
cp ${path to your result}/dnn_best_model/ dnn_best_model_trained
sh infer.sh
<!--
----------------------------------------------------------------
Running inference with num_instance of 64
----------------------------------------------------------------
Inference System time in miliseconds is: 63859
Inference Throughput is
205670
Inference prepare avg is
13.0636
Inference eval avg is
34.5014
-->
```
