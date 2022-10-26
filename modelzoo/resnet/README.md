# Intel Optimized ResNet

## Pre-work
sync submodule code
```
git submodule update --init --recursive
```

apply patch
```
cd modelzoo/resnet
bash patch_resnet.patch
```

Source repo: https://github.com/mlcommons/training_results_v1.0/tree/master/Intel/benchmarks/resnet/2-nodes-16s-8376H-tensorflow

## Model

MicroSoft's [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

## Environment setup

```
cd Dockerfile-ubuntu18.04/
# [option 1]: No proxy required
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow
# [option 2]: Proxy required
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy --build-arg https_proxy
docker run -itd --name resnet --privileged --network host --device=/dev/dri -v $data_path:/home/vmagent/app/dataset/outbrain -v $e2eaiok_path:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-tensorflow /bin/bash
docker exec -it resnet bash
```

## Dataset

https://image-net.org/download-images

## Quick start guide

### Data Processing

Reference https://github.com/mlcommons/training/tree/master/image_classification#3-datasetenvironment

### Training

Edit run_train.sh
```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow
cd mlperf_resnet

echo 3 > /proc/sys/vm/drop_caches 
RANDOM_SEED=`date +%s`
QUALITY=0.757
set -e
export OMP_NUM_THREADS=24
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

MODEL_DIR="./resnet_imagenet_${RANDOM_SEED}"

horovodrun -n 2 HOROVOD_CPU_OPERATIONS=CCL CCL_ATL_TRANSPORT=mpi python imagenet_main.py 1623291220 --train_mode -data_dir /home/vmagent/app/dataset/resnet/ --model_dir $MODEL_DIR --train_epochs 1 --stop_threshold $QUALITY --batch_size 1632 --version 1 --resnet_size 50 --epochs_between_evals 1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee run_train_global_batch_size_3264_${RANDOM_SEED}.log
```
`bash run_train.sh`

### Inference

Edit run_inference.sh
```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow
cd mlperf_resnet

echo 3 > /proc/sys/vm/drop_caches 
RANDOM_SEED=`date +%s`
QUALITY=0.757
set -e
export OMP_NUM_THREADS=24
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

MODEL_DIR="./resnet_imagenet_xxxxxxx"

horovodrun -n 2 HOROVOD_CPU_OPERATIONS=CCL CCL_ATL_TRANSPORT=mpi python imagenet_main.py 1623291220 --data_dir /home/vmagent/app/dataset/resnet/ --model_dir $MODEL_DIR --train_epochs 1 --stop_threshold $QUALITY --batch_size 1632 --version 1 --resnet_size 50 --epochs_between_evals 1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee run_inference_global_batch_size_3264_${RANDOM_SEED}.log
```
`bash run_inference.sh`


### Inference
`bash run_all_in_one.sh`