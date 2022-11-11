# bash run_aiokray_dlrm.sh local_small node_ip
# bash run_aiokray_dlrm.sh distributed_full head_node_ip worker_node_ip...
#!/bin/bash
set -e
seed_num=$(date +%s)
cd ./dlrm
start=$(date +%s)

# check cmd
echo "check cmd"
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
echo "set files path"
hosts_file="../hosts"
config_path="../data_processing/config.yaml"
config_path_infer="../data_processing/config_infer.yaml"
save_path="../data_processing/data_info.txt"
HADOOP_PATH="/home/hadoop-3.3.1"
data_path="/home/vmagent/app/dataset/criteo"
model_path="./result/"
if [ ! -d $model_path ]; then
  mkdir $model_path
fi

# set hosts file
echo "set hosts file"
if [ ! -f $hosts_file ]; then
  touch $hosts_file
fi
echo ${2} > $hosts_file

# set parameters
echo "set parameters"
ncpu_per_proc=1
nproc_per_node=2
ccl_worker_count=4
nnodes=$[ $#-1 ]
world_size=$[ ${nnodes}*${nproc_per_node} ]
num_cpus=$(cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l)
per_cpu_cores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk -F: '{print $2}')
executor_cores=$[ $per_cpu_cores*$num_cpus/$nproc_per_node ]
omp_num_threads=$[ $executor_cores-$ccl_worker_count ]
if [ "${1}" = "local_small" ]; then
    echo "set params for local_small mode"
    train_days="0-3"
    sparse_dense_boundary=1540370
fi
if [ "${1}" = "distributed_full" ]; then
    echo "set params for distributed_full mode" 
    train_days="0-22"
    sparse_dense_boundary=403346
fi
sed -i "s/train_days: \".*\"/train_days: \"${train_days}\"/g" ${config_path}
sed -i "s/train_days: \".*\"/train_days: \"${train_days}\"/g" ${config_path_infer}
sed -i "s/sparse_dense_boundary: [0-9]*/sparse_dense_boundary: ${sparse_dense_boundary}/g" ${config_path}
sed -i "s/sparse_dense_boundary: [0-9]*/sparse_dense_boundary: ${sparse_dense_boundary}/g" ${config_path_infer}
sed -i "s/num_executors: [0-9]*/num_executors: ${world_size}/g" ${config_path}
sed -i "s/num_executors: [0-9]*/num_executors: ${world_size}/g" ${config_path_infer}
sed -i "s/executor_cores: [0-9]*/executor_cores: ${executor_cores}/g" ${config_path}
sed -i "s/executor_cores: [0-9]*/executor_cores: ${executor_cores}/g" ${config_path_infer}
sed -i "s#save_model: \".*\"#save_model: \"${model_path}\"#g" ${config_path}
sed -i "s#load_model: \".*\"#load_model: \"${model_path}\"#g" ${config_path_infer}
sed -i "s#output_folder: \".*\"#output_folder: \"${data_path}\"#g" ${config_path}
sed -i "s#output_folder: \".*\"#output_folder: \"${data_path}\"#g" ${config_path_infer}

# start ray head node
set +e
ray status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "ray has been started"
else
    echo "start ray"
    echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo 1 > /proc/sys/vm/compact_memory; sleep 1
    echo 3 > /proc/sys/vm/drop_caches; sleep 1
    export OMP_NUM_THREADS=${omp_num_threads} && ray start --head --port 5678 --dashboard-host 0.0.0.0 --object-store-memory 268435456000 --system-config='{"object_spilling_threshold":0.98}'
fi
set -e

# data process, cancel this if dataset has been created
echo "Start process dataset"
data_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ../data_processing/convert_to_parquet.py --config_path ${config_path} $dlrm_extra_option 2>&1 | tee run_data_process_${seed_num}.log
# set ray cluster if distributed_full mode is set
if [ "${1}" = "distributed_full" ]; then
    index=1
    for arg in "$@"
    do
        if [ $index \> 2 ]; then
            echo $arg >> $hosts_file
            bash /home/vmagent/app/e2eaiok/scripts/config_passwdless_ssh.sh $args
            ssh $arg export OMP_NUM_THREADS=${executor_cores} && ray start --address="${2}:5678" --object-store-memory 268435456000
        fi
        let index+=1
    done
fi
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ../data_processing/preprocessing.py --config_path ${config_path}  --save_path=${save_path} $dlrm_extra_option 2>&1 | tee -a run_data_process_${seed_num}.log

data_end=$(date +%s)
data_spend=$(( data_end - data_start ))
echo Dataset process time is ${data_spend} seconds.

# model training
echo "start model training"
train_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ./launch.py --distributed --config-path=${config_path} --save-path=${save_path} --ncpu_per_proc=${ncpu_per_proc} --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} --world_size=${world_size} --hostfile ${hosts_file} --master_addr=${2} $dlrm_extra_option 2>&1 | tee run_train_${seed_num}.log

train_end=$(date +%s)
train_spend=$(( train_end - train_start ))
echo training time is ${train_spend} seconds.

# model inference
echo "start model inference"
infer_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ./launch_inference.py --distributed --config-path=${config_path_infer} --save-path=${save_path}  --ncpu_per_proc=${ncpu_per_proc} --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} --world_size=${world_size} --hostfile ${hosts_file} --master_addr=${2} $dlrm_extra_option 2>&1 | tee run_inference_${seed_num}.log

infer_end=$(date +%s)
infer_spend=$(( infer_end - infer_start ))
echo inference time is ${infer_spend} seconds.

end=$(date +%s)
spend=$(( end - start ))
echo Workflow time is ${spend} seconds.