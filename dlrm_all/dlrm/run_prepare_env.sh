# bash run_aiokray_dlrm.sh criteo_small node_ip
# bash run_aiokray_dlrm.sh kaggle node_ip
# bash run_aiokray_dlrm.sh criteo_full head_node_ip worker_node_ip...
#!/bin/bash
set -e
seed_num=$(date +%s)
cd ./dlrm
start=$(date +%s)

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

# check dataset
echo "check dataset"
bash ../run_data_check.sh ${1}

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
executor_cores=6
executor_memory=30
nnodes=$[ $#-1 ]
world_size=$[ ${nnodes}*${nproc_per_node} ]
num_cpus=$(cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l)
per_cpu_cores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk -F: '{print $2}')
omp_num_threads=$[ $per_cpu_cores*$num_cpus/$nproc_per_node-$ccl_worker_count ]
nproc=$(ulimit -u -H)
if [ ${nproc} -le 1048576 ] && [ ${omp_num_threads} -gt 12 ]; then
    omp_num_threads=12
fi

if [ "${1}" = "criteo_small" ]; then
    echo "set params for criteo_small mode"
    train_days="0-3"
    sparse_dense_boundary=1540370
fi

if [ "${1}" = "kaggle" ]; then
    echo "set params for kaggle mode" 
    train_days="24-24"
    sparse_dense_boundary=285147
fi

if [ "${1}" = "criteo_full" ]; then
    echo "set params for criteo_full mode" 
    train_days="0-22"
    sparse_dense_boundary=403346
fi

# start ray
set +e
ray status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "OMP_NUM_THREADS: ${omp_num_threads}"
    export OMP_NUM_THREADS=${omp_num_threads}
    echo "ray has been started"
else
    echo "start ray"
    echo "OMP_NUM_THREADS: ${omp_num_threads}"
    echo "clean memory"
    echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
    echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
    echo 1 > /proc/sys/vm/compact_memory; sleep 1
    echo 3 > /proc/sys/vm/drop_caches; sleep 1
    need_memory_criteo_small=171798691840
    need_memory_kaggle=10737418240
    unit_memory=42949672960
    avail_memory=$[ $[ $(cat /proc/meminfo | grep MemAvailable | awk -F' ' '{print $2}')*1024 ]-$unit_memory]
    if [ $avail_memory -gt $need_memory_criteo_small ]; then
        obj_memory=$need_memory_criteo_small
    elif [ $avail_memory -gt $need_memory_kaggle ]; then
        obj_memory=$avail_memory
        echo "WARNING: Memory is not enough for 'criteo_small' mode, may cause Ray object spilling. Please use 'kaggle' mode."
    else
        echo "Error: Please make sure the available memory is at least greater than 50G, exit"
        exit
    fi
    echo object-store-memory is $[ $obj_memory/1024/1024/1024 ] GB
    export OMP_NUM_THREADS=${omp_num_threads} && ray start --node-ip-address="${2}" --head --port 5678 --dashboard-host 0.0.0.0 --object-store-memory $obj_memory --system-config='{"object_spilling_threshold":0.98}'
fi

retry=0
while [ -z "$memory" ]; do
    if [ $retry -gt 5 ]; then
        echo "unable to start ray, exit"
        exit
    fi
    echo "wait 3 secs for ray to start"
    sleep 3
    memory=`ray status | grep memory | head -1 | sed "s#[0-9]*.[0-9]*/\([0-9]*\).[0-9]* GiB memory#\1#g"`
    echo memory is $memory GB
    retry=$[ $retry + 1 ]
done
memory_executor=$[ $memory / $executor_memory ]
num_cpus_executor=$[ $per_cpu_cores*$num_cpus/$executor_cores ]
num_executors=$memory_executor
if [ $memory_executor -gt $num_cpus_executor ]; then
    num_executors=$num_cpus_executor
fi
echo num_executors is $num_executors

sed -i "s/train_days: \".*\"/train_days: \"${train_days}\"/g" ${config_path}
sed -i "s/train_days: \".*\"/train_days: \"${train_days}\"/g" ${config_path_infer}
sed -i "s/sparse_dense_boundary: [0-9]*/sparse_dense_boundary: ${sparse_dense_boundary}/g" ${config_path}
sed -i "s/sparse_dense_boundary: [0-9]*/sparse_dense_boundary: ${sparse_dense_boundary}/g" ${config_path_infer}
sed -i "s/num_executors: [0-9]*/num_executors: ${num_executors}/g" ${config_path}
sed -i "s/num_executors: [0-9]*/num_executors: ${num_executors}/g" ${config_path_infer}
sed -i "s#save_model: \".*\"#save_model: \"${model_path}\"#g" ${config_path}
sed -i "s#load_model: \".*\"#load_model: \"${model_path}\"#g" ${config_path_infer}
sed -i "s#output_folder: \".*\"#output_folder: \"${data_path}\"#g" ${config_path}
sed -i "s#output_folder: \".*\"#output_folder: \"${data_path}\"#g" ${config_path_infer}
