# source ~/.local/env/setvars.sh
set -ex
seed_num=$(date +%s)

start=$(date +%s)
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

cd ./dlrm
# data process, cancel this if dataset has been created
echo "Start process dataset"
data_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python ../data_processing/convert_to_parquet.py --config_path "../data_processing/config.yaml"
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python ../data_processing/preprocessing.py --config_path "../data_processing/config.yaml"  --save_path="../data_processing/data_info.txt"
data_end=$(date +%s)
data_spend=$(( data_end - data_start ))
echo Dataset process time is ${data_spend} seconds.

# make dataset path
HADOOP_PATH="/home/hadoop-3.3.1"
output_path="hdfs://10.1.8.4:9000/mnt/DP_disk3/output"
if [[ "${output_path}" =~ ^hdfs.* ]]
then
    echo "make hdfs path"
    ${HADOOP_PATH}/bin/hdfs dfs -mkdir ${output_path}/train
    ${HADOOP_PATH}/bin/hdfs dfs -mv ${output_path}/dlrm_categorified_day_* ${output_path}/train/
else
    echo "make local path"
    mkdir ${output_path}/train
    mv ${output_path}/dlrm_categorified_day_* ${output_path}/train/
fi

# clean memory
echo "clean memory"
clean_start=$(date +%s)
parallel-ssh -i -h ../hosts "echo never > /sys/kernel/mm/transparent_hugepage/enabled"
parallel-ssh -i -h ../hosts sleep 1
parallel-ssh -i -h ../hosts "echo never > /sys/kernel/mm/transparent_hugepage/defrag"
parallel-ssh -i -h ../hosts sleep 1
parallel-ssh -i -h ../hosts "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
parallel-ssh -i -h ../hosts sleep 1
parallel-ssh -i -h ../hosts "echo always > /sys/kernel/mm/transparent_hugepage/defrag"
parallel-ssh -i -h ../hosts sleep 1
parallel-ssh -i -h ../hosts "echo 1 > /proc/sys/vm/compact_memory"
parallel-ssh -i -h ../hosts sleep 1
parallel-ssh -i -h ../hosts "echo 3 > /proc/sys/vm/drop_caches"
parallel-ssh -i -h ../hosts sleep 1
clean_end=$(date +%s)
clean_spend=$(( clean_end - clean_start ))
echo Clean memory time is ${clean_spend} seconds.

end=$(date +%s)
spend=$(( end - start ))
echo Workflow time is ${spend} seconds.
