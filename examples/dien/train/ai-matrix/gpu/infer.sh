#!/bin/bash

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=1
# export LD_PRELOAD=/home2/yunfeima/anaconda3/envs/tc/lib/libtcmalloc.so.4

TOTAL_RECOMMDS=606720

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

# batchs='256 512 1024'
batchs='128'

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
	# numactl -N 0 python script/train.py --mode=test --batch_size=$batch |& tee results/result_infer_${batch}.txt
        numactl -C 0-0,40-40 --membind=0 python script/train.py --mode=test --batch_size=$batch --num-inter-threads=20 --num-intra-threads=20 |& tee results/result_infer_${batch}.txt
	end=`date +%s%N`
	total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$TOTAL_RECOMMDS
    system_performance=$((1000*$total_images/$total_time))
    echo "Total recommendations: $total_images" >> results/result_infer_${batch}.txt
    echo "System time in miliseconds is: $total_time" >> results/result_infer_${batch}.txt
    echo "System performance in recommendations/second is: $system_performance" >> results/result_infer_${batch}.txt
done

python process_results.py --infer
