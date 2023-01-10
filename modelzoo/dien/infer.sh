#!/bin/bash

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=1

batchs='128'
NUM_INSTANCES=64

rm -r results
mkdir -p results

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running inference with num_instance of $NUM_INSTANCES"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    NUM_INSTANCES_BEGIN=0
    NUM_INSTANCES_END=${NUM_INSTANCES}
    NUM_INSTANCES_MINUS_ONE=$((${NUM_INSTANCES}-1))
    OUT_SHIFT=0
    CORES_PER_INST=1
    bash run_infer.sh ${NUM_INSTANCES_BEGIN} ${NUM_INSTANCES_END} ${OUT_SHIFT}

    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    sleep 3
    echo "Inference System time in miliseconds is: $total_time" | tee -a results/result_infer_${batch}.txt
done

# For inference result
echo 'Inference Throughput is '; grep performance -r ./ | awk '{sum+=$NF}END{print sum}'
echo 'Inference prepare avg is '; grep "time breakdown" -r ./ | awk '{sum+=$7}END{print sum/NR}'
echo 'Inference eval avg is '; grep "time breakdown" -r ./ | awk '{sum+=$11}END{print sum/NR}'

