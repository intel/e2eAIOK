#!/bin/bash

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=1
# export MKLDNN_VERBOSE=2
# export TF_DUMP_GRAPH_PREFIX=/home2/yunfeima/tmp/dien/dump_graph
# export TF_XLA_FLAGS="--tf_xla_clustering_debug"

# export MKL_VERBOSE=1

TOTAL_RECOMMDS=606720

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

# batchs='256 512 1024'
batchs='128'
# pb="/home2/yunfeima/tmp/dien/constant_sub_node_graph_noparallel.pb"
pb="/home2/yunfeima/tmp/dien/constant_sub_node_graph_acconly.pb"
# pb="/home2/yunfeima/tmp/dien/constant_sub_node_fixed_reshape.pb"

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
    numactl -C 0-0,40-40 --membind=0 python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-1_1024_notcmalloc_manual_g115.log
	# numactl -C 0-19,40-59 --membind=0 python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=20 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-1_1024_notcmalloc_manual_g115.log &
    # numactl -C 20-39,60-79 --membind=1 python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=20 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_-2_1024_notcmalloc_manual_g115.log &
    
    wait
	# end=`date +%s%N`
	# total_time=$(((end-start)/1000000))
    # #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    # total_images=$TOTAL_RECOMMDS
    # system_performance=$((1000*$total_images/$total_time))
    # echo "Total recommendations: $total_images" >> results/result_infer_${batch}.txt
    # echo "System time in miliseconds is: $total_time" >> results/result_infer_${batch}.txt
    # echo "System performance in recommendations/second is: $system_performance" >> results/result_infer_${batch}.txt
done

# python process_results.py --infer
