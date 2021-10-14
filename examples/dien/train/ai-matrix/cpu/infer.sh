#!/bin/bash

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=1
# export LD_PRELOAD=/home2/yunfeima/anaconda3/envs/tc/lib/libtcmalloc.so.4

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

# batchs='256 512 1024'
#batchs='128 256 512 1024'
batchs='128'

echo "----------------------------------------------------------------"
echo "Running inference preprocessing"
echo "----------------------------------------------------------------"

python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_inference.py | tee results/preprocessing_for_inference_log.txt
if [ $? != 0 ]; then
    exit
fi

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running inference with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    NUM_INSTANCES=64
    NUM_INSTANCES_MINUS_ONE=$((${NUM_INSTANCES}-1))
    CORES_PER_INST=1
    for (( i = 0; i < ${NUM_INSTANCES_MINUS_ONE}; i++ ))
    do
        numactl -C$((${i}*${CORES_PER_INST}))-$((${CORES_PER_INST}*${i}+${CORES_PER_INST}-1)) --localalloc python script/train.py --mode=test --advanced --slice_id=${i} --batch_size=$batch --num-inter-threads=1 --num-intra-threads=1 2>/dev/null | tee results/result_infer_${batch}_${i}.txt &
    done
    numactl -C$((${NUM_INSTANCES_MINUS_ONE}*${CORES_PER_INST}))-$((${CORES_PER_INST}*${NUM_INSTANCES_MINUS_ONE}+${CORES_PER_INST}-1)) --localalloc python script/train.py --mode=test --advanced --slice_id=${NUM_INSTANCES_MINUS_ONE} --batch_size=$batch --num-inter-threads=1 --num-intra-threads=1 2>results/result_infer_${batch}_${NUM_INSTANCES_MINUS_ONE}_err.txt | tee results/result_infer_${batch}_${NUM_INSTANCES_MINUS_ONE}.txt
    if [ $? != 0 ]; then
	echo "Inference failed! Trace log:"
	cat results/result_infer_${batch}_${NUM_INSTANCES_MINUS_ONE}_err.txt
        exit
    fi

    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    sleep 3
    echo "Inference System time in miliseconds is: $total_time" | tee -a results/result_infer_${batch}.txt
done

#python process_results.py --infer
