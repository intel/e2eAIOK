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
```

---
# Evaluate DIEN solution

## option 1. Step By Step guide

### Data Process
```
# Now you are running inside docker, /home/vmagent/app/
source /etc/profile.d/spark-env.sh
pip install pyrecdp
sh /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/start_spark_service.sh 
python /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/preprocessing.py --train
python /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/preprocessing.py --test
python /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/preprocessing.py --inference
```

### Training
```
# test our best parameter
python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien --no_sigopt

# optional: use SDA to search best parameter
SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien
```

## option 2. Run All in one script
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
    training_time: 4474.986137151718
=============================================== 
-->
```

## Inference
```
rm /home/vmagent/app/e2eaiok/modelzoo/dien/train/ai-matrix/dnn_best_model_trained/ -rf
cp ${path to your result}/dnn_best_model/ dnn_best_model_trained
cd /home/vmagent/app/e2eaiok/modelzoo/dien/train/ai-matrix/
./infer.sh
<!--
test_auc: 0.8234 ----test_loss: 1.2291 ---- test_accuracy: 0.750387738 ---- test_aux_loss: 1.0837 ---- eval_time: 34.719 ---- prepare_time: 19.282
Total recommendations: 121216
Approximate accelerator time in seconds is 34.719
Approximate accelerator performance in recommendations/second is 3491.335
Process time breakdown, prepare data took 19.282 and test took 34.719, avg is prepare 3.856, test 6.944
-->
```