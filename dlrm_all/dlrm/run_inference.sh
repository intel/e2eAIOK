# bash run_inference.sh local_small node_ip
# bash run_inference.sh distributed_full head_node_ip worker_node_ip...
#!/bin/bash
set -e
seed_num=$(date +%s)

# check cmd
echo "check cmd"
if [ "${2}" = "" ]; then
    echo "error: node_ip is None"
fi

if [[ ${1} != "local_small" && ${1} != "distributed_full" ]]; then
    echo "error: need to use 'local_small' or 'distributed_full' mode"
    exit
fi

index=1
for arg in "$@"
    do
        if [ $index \> 1 ]; then
            if [[ ! $arg =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                echo "error: node_ip<$arg> is invalid"
                exit
            fi
            field1=$(echo $arg|cut -d. -f1)
            field2=$(echo $arg|cut -d. -f2)
            field3=$(echo $arg|cut -d. -f3)
            field4=$(echo $arg|cut -d. -f4)
            if [[ $field1 -gt 255 || $field2 -gt 255 || $field3 -gt 255 || $field4 -gt 255 ]]; then
                echo "error: node_ip<$field1.$field2.$field3.$field4> is invalid"
                exit
            fi
        fi
        let index+=1
    done 

# set files path
hosts_file="../hosts"
config_path_infer="../data_processing/config_infer.yaml"
save_path="../data_processing/data_info.txt"

# set parameters
ncpu_per_proc=1
nproc_per_node=2
ccl_worker_count=4
nnodes=$[ $#-1 ]
world_size=$[ ${nnodes}*${nproc_per_node} ]
num_cpus=$(cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l)
per_cpu_cores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk -F: '{print $2}')
executor_cores=$[ $per_cpu_cores*$num_cpus/$nproc_per_node ]
omp_num_threads=$[ $executor_cores-$ccl_worker_count ]

# check ray
set +e
ray status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "OMP_NUM_THREADS: ${omp_num_threads}"
    export OMP_NUM_THREADS=${omp_num_threads}
    echo "ray has been started."
else
    echo "start ray."
    echo "OMP_NUM_THREADS: ${omp_num_threads}"
    echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo 1 > /proc/sys/vm/compact_memory; sleep 1
    echo 3 > /proc/sys/vm/drop_caches; sleep 1
    export OMP_NUM_THREADS=${omp_num_threads} && ray start --head --port 5678 --dashboard-host 0.0.0.0 --object-store-memory 268435456000 --system-config='{"object_spilling_threshold":0.98}'
fi

# model inference
echo "start model inference"
cd ./dlrm
infer_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ./launch_inference.py --distributed --config-path=${config_path_infer} --save-path=${save_path}  --ncpu_per_proc=${ncpu_per_proc} --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} --world_size=${world_size} --hostfile ${hosts_file} --master_addr=${2} $dlrm_extra_option 2>&1 | tee run_inference_${seed_num}.log

infer_end=$(date +%s)
infer_spend=$(( infer_end - infer_start ))
echo inference time is ${infer_spend} seconds.