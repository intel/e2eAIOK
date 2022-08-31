# Intel Optimized Wide and Deep

## Pre-work
sync submodule code
```
git submodule update --init --recursive
```

apply patch
```
cd modelzoo/WnD/TensorFlow2
bash patch_wnd.patch
```

Source repo: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep

## Model

Google's [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

## Environment setup

```
cd Dockerfile-ubuntu18.04/
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow
docker run -it --privileged --network host --device=/dev/dri -v $data_path:/home/vmagent/app/dataset/outbrain -v $e2eaiok_path:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ e2eaiok-tensorflow /bin/bash
```

## Dataset

The original dataset can be downloaded at https://www.kaggle.com/c/outbrain-click-prediction/data

## Quick start guide

### Data preprocessing
```
# start spark service
# need to setup ssh service first
cd /home/vmagent/app/e2eaiok/modelzoo/WnD/TensorFlow2/data_processing/
cp spark-defaults.conf /home/spark-3.2.1-bin-hadoop3.2/conf/
mkdir -p /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/spark_local_dir
mkdir -p /home/mnt/applicationHistory
sh ./start_spark_service.sh

# data preprocess with spark
bash scripts/spark_preproc.sh
```

### Training

`bash scripts/train.sh`

### Inference

`bash scripts/inference.sh`
