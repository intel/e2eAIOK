#!/bin/bash
# export LD_PRELOAD=/home2/yunfeima/downloads/gperftools-2.7/.libs/libtcmalloc.so.4
export LD_PRELOAD=/home2/yunfeima/anaconda3/envs/tc/lib/libtcmalloc.so.4
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=1
export KMP_BLOCKTIME=1
# export MKLDNN_VERBOSE=1

# export TF_DISABLE_MKL=1

TOTAL_RECOMMDS=606720

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

# batchs='256 512 1024'
batchs='128'
# pb="/home2/yunfeima/tmp/dien/constant_sub_node_graph.pb"
pb="/home2/yunfeima/tmp/dien/constant_sub_node_graph_acconly.pb"

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
	numactl -C 0-0 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_0_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 1-1 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_1_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 2-2 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_2_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 3-3 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_3_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 4-4 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_4_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 5-5 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_5_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 6-6 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_6_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 7-7 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_7_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 8-8 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_8_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 9-9 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_9_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 10-10 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_10_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 11-11 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_11_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 12-12 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_12_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 13-13 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_13_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 14-14 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_14_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 15-15 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_15_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 16-16 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_16_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 17-17 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_17_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 18-18 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_18_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 19-19 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_19_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 20-20 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_20_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 21-21 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_21_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 22-22 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_22_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 23-23 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_23_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 24-24 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_24_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 25-25 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_25_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 26-26 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_26_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 27-27 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_27_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 28-28 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_28_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 29-29 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_29_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 30-30 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_30_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 31-31 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_31_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 32-32 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_32_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 33-33 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_33_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 34-34 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_34_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 35-35 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_35_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 36-36 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_36_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 37-37 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_37_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 38-38 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_38_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 39-39 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_39_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 40-40 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_40_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 41-41 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_41_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 42-42 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_42_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 43-43 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_43_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 44-44 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_44_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 45-45 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_45_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 46-46 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_46_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 47-47 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_47_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 48-48 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_48_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 49-49 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_49_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 50-50 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_50_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 51-51 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_51_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 52-52 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_52_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 53-53 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_53_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 54-54 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_54_128_tcmalloc_intel-maint-tf-1216.log & 
    numactl -C 55-55 -l python script/inference_pb.py         --mode=test         --batch_size=$batch         --pb_path=$pb         --num-inter-threads=1         --num-intra-threads=1 2>&1 | tee /home2/yunfeima/logs/dien/multi-instance/1_55_128_tcmalloc_intel-maint-tf-1216.log &     
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
