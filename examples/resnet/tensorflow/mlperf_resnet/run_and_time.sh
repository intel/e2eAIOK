#/usr/bin/bash

echo 3 > /proc/sys/vm/drop_caches 

RANDOM_SEED=`date +%s`

QUALITY=0.759

set -e

# Register the model as a source root


# MLPerf


echo $PYTHONPATH

mkdir -p ./IntelMLPerf/

MODEL_DIR="./IntelMLPerf/resnet_imagenet_${RANDOM_SEED}"

export OMP_NUM_THREADS=12

export KMP_BLOCKTIME=1
nohup mpirun --allow-run-as-root -mca btl_tcp_if_include eth2 -np 32 -H sr231:16,sr232:16  ~/sw/miniconda3/envs/hvd/bin/python -u /home/xianyang/BlueWhale-poc/resnet/tensorflow/mlperf_resnet/imagenet_main.py $RANDOM_SEED --data_dir /mnt/DP_disk3/imagenet/models/research/slim/datasets/ILSVRC2012  --model_dir $MODEL_DIR --train_epochs 1 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002 --num_gpus=0 --data_format='channels_last' >./two_node_32_worker.log 2>&1 &

