#/usr/bin/bash
source /opt/intel/oneapi/setvars.sh --force
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


horovodrun -n 2 HOROVOD_CPU_OPERATIONS=CCL CCL_ATL_TRANSPORT=mpi python imagenet_main.py 1623291220 --data_dir /home/vmagent/app/dataset/resnet/ --model_dir $MODEL_DIR --train_epochs 1 --stop_threshold $QUALITY --batch_size 1632 --version 1 --resnet_size 50 --epochs_between_evals 1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee run_train_global_batch_size_3264_${RANDOM_SEED}.log