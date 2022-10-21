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
# [option 1]: No proxy required
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow
# [option 2]: Proxy required
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy --build-arg https_proxy
docker run -itd --name wnd --privileged --network host --device=/dev/dri -v $data_path:/home/vmagent/app/dataset/outbrain -v $e2eaiok_path:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-tensorflow /bin/bash
docker exec -it wnd bash
```

## Dataset

The original dataset can be downloaded at https://www.kaggle.com/c/outbrain-click-prediction/data

## Quick start guide

### Data preprocessing
```
# start spark service
# need to setup ssh service first
cd /home/vmagent/app/e2eaiok/conf/spark/
cp spark-defaults.conf /home/spark-3.2.1-bin-hadoop3.2/conf/
mkdir -p /home/vmagent/app/e2eaiok/spark_data_processing/spark_local_dir
mkdir -p /home/mnt/applicationHistory
sh ./start_spark_service.sh

# data preprocess with spark
bash scripts/spark_preproc.sh
```

### Training

Edit scripts/train.sh
```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow

export OMP_NUM_THREADS=30

time horovodrun -np 8 -H ${node1}:2,${node2}:2,${node3}:2,${node4}:2 --network-interface ${interface} \
/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python -u main.py \
  --train_data_pattern '/home/vmagent/app/dataset/outbrain/train/part*' \
  --eval_data_pattern '/home/vmagent/app/dataset/outbrain/valid/part*' \
  --model_dir ./checkpoints --results_dir ./ \
  --dataset_meta_file data/outbrain/outbrain_meta.yaml \
  --global_batch_size 524288 \
  --eval_batch_size 524288 \
  --num_epochs 20 \
  --deep_learning_rate 0.011150920451008404 \
  --linear_learning_rate 1 \
  --deep_hidden_units 128 128 64 \
  --metric MAP \
  --deep_warmup_epochs 3 --deep_dropout 0.21076275738334013 \
  --metric_threshold 0.6553
```
`bash scripts/train.sh`

### Inference

Edit scripts/inference.sh
```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow

export OMP_NUM_THREADS=30

time horovodrun -np 8 -H ${node1}:2,${node2}:2,${node3}:2,${node4}:2 --network-interface ${interface} \
/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python -u main.py \
  --train_data_pattern '/home/vmagent/app/dataset/outbrain/train/part*' \
  --eval_data_pattern '/home/vmagent/app/dataset/outbrain/valid/part*' \
  --model_dir ./checkpoints \
  --dataset_meta_file data/outbrain/outbrain_meta.yaml \
  --deep_learning_rate 0.00048 \
  --linear_learning_rate 0.8 \
  --eval_batch_size 1048576 \
  --evaluate \
  --use_checkpoint \
  --benchmark \
  --benchmark_warmup_steps 50 \
  --benchmark_steps 100 \
  --metric MAP --deep_hidden_units 128 128 64
```
`bash scripts/inference.sh`
