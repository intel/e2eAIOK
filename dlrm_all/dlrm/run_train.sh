# bash run_train.sh local_small node_ip
# bash run_train.sh distributed_full head_node_ip worker_node_ip...
#!/bin/bash
set -e
seed_num=$(date +%s)

ncpu_per_proc=1
nproc_per_node=2
ccl_worker_count=4
nnodes=$[ $#-1 ]
world_size=$[ ${nnodes}*${nproc_per_node} ]
num_cpus=$(cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l)
per_cpu_cores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk -F: '{print $2}')
executor_cores=$[ $per_cpu_cores*$num_cpus/$nproc_per_node ]
omp_num_threads=$[ $executor_cores-$ccl_worker_count ]
export OMP_NUM_THREADS=${omp_num_threads}

hosts_file="../hosts"
config_path="../data_processing/config.yaml"
save_path="../data_processing/data_info.txt"

echo "start model training"
cd ./dlrm
train_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ./launch.py --distributed --config-path=${config_path} --save-path=${save_path} --ncpu_per_proc=${ncpu_per_proc} --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} --world_size=${world_size} --hostfile ${hosts_file} --master_addr=${2} $dlrm_extra_option 2>&1 | tee run_train_${seed_num}.log
train_end=$(date +%s)
train_spend=$(( train_end - train_start ))
echo training time is ${train_spend} seconds.