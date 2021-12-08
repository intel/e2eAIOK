#!/bin/bash
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=4
export HOROVOD_CPU_OPERATIONS=CCL
# export MKLDNN_VERBOSE=2
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY="0,32"
export HOROVOD_THREAD_AFFINITY="1,33"
#export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="0,1,32,33"


NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"
TOTAL_RECOMMDS=512000

rm -r dnn_save_path dnn_best_model
mkdir dnn_save_path dnn_best_model

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='256 512 1024'
#batchs='1024'

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running training with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
        time mpirun -n 1 -hosts 172.16.8.30 -ppn 1 -iface ens21f1 -print-rank-map -prepend-rank -verbose python /home/xxx/dien/script/train.py --mode=train --advanced --batch_size=$batch --num-inter-threads=4 --num-intra-threads=32 |& tee /home/xxx/dien/results/result_train_${batch}.txt
        #time mpirun -n 2 -hosts 172.16.8.30,172.16.8.19 -ppn 1 -iface ens21f1 -print-rank-map -prepend-rank -verbose python /home/xxx/dien/script/train.py --mode=train --advanced --batch_size=$batch --num-inter-threads=4 --num-intra-threads=32 |& tee /home/xxx/dien/results/result_train_${batch}.txt
        #time mpirun -n 4 -hosts 172.16.8.30,172.16.8.19,172.16.8.24,172.16.8.27 -ppn 1 -iface ens21f1 -print-rank-map -prepend-rank -verbose python /home/xxx/dien/script/train.py --mode=train --advanced --batch_size=$batch --num-inter-threads=4 --num-intra-threads=32 |& tee /home/xxx/dien/results/result_train_${batch}.txt
	#time horovodrun --timeline-filename /home/xxx/dien/horovod_timeline_2.json -np 2 -H 10.1.0.19:1,10.1.0.30:1 --network-interface enp134s0f1 --verbose python /home/xxx/dien/script/train.py --mode=train --advanced --batch_size=$batch --num-inter-threads=4 --num-intra-threads=24 |& tee /home/xxx/dien/results/result_train_${batch}.txt
	end=`date +%s%N`
	total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$TOTAL_RECOMMDS
    system_performance=$((1000*$total_images/$total_time))
    echo "Total recommendations: $total_images" >> results/result_train_${batch}.txt
    echo "System time in miliseconds is: $total_time" >> results/result_train_${batch}.txt
    echo "System performance in recommendations/second is: $system_performance" >> results/result_train_${batch}.txt
done

#python process_results.py --train
