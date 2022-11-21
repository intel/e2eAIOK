# Intel® End-to-End AI Optimization Kit for Resnet
## Original source disclose
Source repo: https://github.com/mlcommons/training_results_v1.0/tree/master/Intel/benchmarks/resnet/2-nodes-16s-8376H-tensorflow

Notes: ResNet training is based on ImageNet1K dataset, we evaluated Top-1 Accuracy with stock model (based on MLPerf submission) at ImageNet1K dataset, and final Top-1 Accuracy is 0.757

public reference paper: https://arxiv.org/abs/1512.03385

---

# Quick Start
## Enviroment Setup
``` bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b tensorflow -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```
## Enter Docker
```
sshpass -p docker ssh ${host0} -p 12344
```

## Workflow Prepare
``` bash
# prepare model codes
cd /home/vmagent/app/e2eaiok/modelzoo/resnet
bash patch_resnet.patch

# Download Dataset
The TF ResNet50-v1.5 model is trained with ImageNet 1K, a popular image classification dataset from ILSVRC challenge. The dataset can be downloaded from:

http://image-net.org/download-images

More dataset requirements can be found at:

https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment
```

## Data Processing
```
Reference https://github.com/mlcommons/training/tree/master/image_classification#3-datasetenvironment
```

## Training
```
# systemg
echo 3 > /proc/sys/vm/drop_caches 
RANDOM_SEED=`date +%s`
QUALITY=0.757
export OMP_NUM_THREADS=24
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

cd /home/vmagent/app/e2eaiok/; python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/resnet --model_name resnet --conf /home/vmagent/app/e2eaiok/tests/cicd/conf/e2eaiok_defaults_resnet_example.conf 
```

## Inference
```
# edit run_inference.sh
echo 3 > /proc/sys/vm/drop_caches 
RANDOM_SEED=`date +%s`
QUALITY=0.757
export OMP_NUM_THREADS=24
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

MODEL_DIR="./resnet_imagenet_xxxxxxx"

horovodrun -n 2 HOROVOD_CPU_OPERATIONS=CCL CCL_ATL_TRANSPORT=mpi python imagenet_main.py 1623291220 --eval_mode --data_dir /home/vmagent/app/dataset/resnet/ --model_dir $MODEL_DIR --train_epochs 1 --stop_threshold $QUALITY --batch_size 1632 --version 1 --resnet_size 50 --epochs_between_evals 1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee run_inference_global_batch_size_3264_${RANDOM_SEED}.log
```
`bash run_inference.sh`
