# bash run_aiokray_dlrm.sh criteo_small node_ip
# bash run_aiokray_dlrm.sh kaggle node_ip
# bash run_aiokray_dlrm.sh criteo_full head_node_ip worker_node_ip...
#!/bin/bash
set -eo pipefail
seed_num=$(date +%s)

# check cmd
echo "check cmd"
if [ "${2}" = "" ]; then
    echo "error: node_ip is None"
fi

if [[ ${1} != "criteo_small" && ${1} != "criteo_full" && ${1} != "kaggle" ]]; then
    echo "error: need to use 'criteo_small' or 'criteo_full' or 'kaggle' mode"
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
config_path="../data_processing/config.yaml"
save_path="../data_processing/data_info.txt"

# set parameters
nproc_per_node=2
ccl_worker_count=4
nnodes=$[ $#-1 ]
world_size=$[ ${nnodes}*${nproc_per_node} ]
num_cpus=$(cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l)
per_cpu_cores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk -F: '{print $2}')
omp_num_threads=$[ $per_cpu_cores*$num_cpus/$nproc_per_node-$ccl_worker_count ]
nproc=$(ulimit -u -H)
if [ ${nproc} -le 1048576 ] && [ ${omp_num_threads} -gt 12 ]; then
    omp_num_threads=12
fi

# check ray
set +e
ray status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "OMP_NUM_THREADS: ${omp_num_threads}"
    export OMP_NUM_THREADS=${omp_num_threads}
    echo "ray has been started"
else
    echo "start ray"
    echo "OMP_NUM_THREADS: ${omp_num_threads}"
    echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo 1 > /proc/sys/vm/compact_memory; sleep 1
    echo 3 > /proc/sys/vm/drop_caches; sleep 1
    export OMP_NUM_THREADS=${omp_num_threads} && ray start --node-ip-address="${2}" --head --port 5678 --dashboard-host 0.0.0.0 --object-store-memory 171798691840 --system-config='{"object_spilling_threshold":0.98}'
fi

# data process
echo "Start process dataset"
cd ./dlrm
data_start=$(date +%s)
data_path_train="/home/vmagent/app/dataset/criteo/train"
if [ ! -d $data_path_train ]; then
  rm -rf $data_path_train
fi
rm -rf /home/vmagent/app/dataset/criteo/dlrm_*
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ../data_processing/convert_to_parquet.py --config_path ${config_path} --run_mode=$1 $dlrm_extra_option 2>&1 | tee run_data_process_${seed_num}.log
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ../data_processing/preprocessing.py --config_path ${config_path}  --save_path=${save_path} $dlrm_extra_option 2>&1 | tee -a run_data_process_${seed_num}.log

data_end=$(date +%s)
data_spend=$(( data_end - data_start ))
echo Dataset process time is ${data_spend} seconds.
