#!/bin/bash

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=6

# export MKLDNN_VERBOSE=2

TOTAL_RECOMMDS=606720

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

# batchs='256 512 1024'
batchs='128'
# pb="/home2/yunfeima/tmp/dien/constant_sub_node_graph_noparallel.pb"
pb="/home2/yunfeima/tmp/dien/constant_sub_node_graph_acconly.pb"

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
    # numactl -N 0 --localalloc python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb          2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-1_1024_notcmalloc_manual_g115.log 
    numactl -l -N 0  python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb      --num-inter-threads=4  --num-intra-threads=24   2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-1_1024_notcmalloc_manual_g115.log 
    # numactl -C 0-19,40-59 python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb    --num-inter-threads=20         --num-intra-threads=20      2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-1_1024_notcmalloc_manual_g115.log
    # numactl -C 0-19,40-59 python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb    --num-inter-threads=40         --num-intra-threads=40      2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-1_1024_notcmalloc_manual_g115.log
done

# python process_results.py --infer
