#!/bin/bash

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=1
# export LD_PRELOAD=/home2/yunfeima/anaconda3/envs/tc/lib/libtcmalloc.so.4

#if [ -d results ]; then
#    mv results results_$(date +%Y%m%d%H%M%S)
#fi
#mkdir results
#ssh -p 12345 dr8s19 mkdir -p /home/xxx/dien/results

# batchs='256 512 1024'
#batchs='128 256 512 1024'
batchs='128'

#echo "----------------------------------------------------------------"
#echo "Running inference preprocessing"
#echo "----------------------------------------------------------------"
#
#python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_inference.py | tee results/preprocessing_for_inference_log.txt
#if [ $? != 0 ]; then
#    exit
#fi

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running inference with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    NUM_INSTANCES=64
    NUM_INSTANCES_BEGIN=0
    NUM_INSTANCES_END=${NUM_INSTANCES}
    NUM_INSTANCES_MINUS_ONE=$((${NUM_INSTANCES}-1))
    OUT_SHIFT=0
    CORES_PER_INST=1
    bash run_infer.sh ${NUM_INSTANCES_BEGIN} ${NUM_INSTANCES_END} ${OUT_SHIFT}
    #ssh -p 12345 dr8s30 "cd /home/xxx/dien/; bash run_infer.sh ${NUM_INSTANCES_BEGIN} ${NUM_INSTANCES_END} ${OUT_SHIFT}"
    #NUM_INSTANCES_BEGIN=$(( ${NUM_INSTANCES_END} ))
    #NUM_INSTANCES_END=$(( ${NUM_INSTANCES_BEGIN} + ${NUM_INSTANCES} ))
    #OUT_SHIFT=$((${OUT_SHIFT} + ${NUM_INSTANCES}))
    #ssh -p 12345 dr8s19 "cd /home/xxx/dien/; bash run_infer.sh  ${NUM_INSTANCES_BEGIN} ${NUM_INSTANCES_END} ${OUT_SHIFT}"
    #NUM_INSTANCES_BEGIN=$(( ${NUM_INSTANCES_END} ))
    #NUM_INSTANCES_END=$(( ${NUM_INSTANCES_BEGIN} + ${NUM_INSTANCES} ))
    #OUT_SHIFT=$((${OUT_SHIFT} + ${NUM_INSTANCES}))
    #ssh -p 12345 dr8s24 "cd /home/xxx/dien/; bash run_infer.sh  ${NUM_INSTANCES_BEGIN} ${NUM_INSTANCES_END} ${OUT_SHIFT}"
    #NUM_INSTANCES_BEGIN=$(( ${NUM_INSTANCES_END} ))
    #NUM_INSTANCES_END=$(( ${NUM_INSTANCES_BEGIN} + ${NUM_INSTANCES} ))
    #OUT_SHIFT=$((${OUT_SHIFT} + ${NUM_INSTANCES}))
    #ssh -p 12345 dr8s27 "cd /home/xxx/dien/; bash run_infer.sh  ${NUM_INSTANCES_BEGIN} ${NUM_INSTANCES_END} ${OUT_SHIFT}"
    # Wait for all parallel jobs to finish
    #while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done
    sleep 20

    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    sleep 3
    echo "Inference System time in miliseconds is: $total_time" | tee -a results/result_infer_${batch}.txt
done

#python process_results.py --infer
