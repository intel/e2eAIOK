# bash run_data_process.sh local_small node_ip
# bash run_data_process.sh distributed_full head_node_ip worker_node_ip...
#!/bin/bash
set -e
seed_num=$(date +%s)

config_path="../data_processing/config.yaml"
save_path="../data_processing/data_info.txt"

echo "Start process dataset"
cd ./dlrm
data_start=$(date +%s)
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ../data_processing/convert_to_parquet.py --config_path ${config_path} $dlrm_extra_option 2>&1 | tee run_data_process_${seed_num}.log
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ../data_processing/preprocessing.py --config_path ${config_path}  --save_path=${save_path} $dlrm_extra_option 2>&1 | tee -a run_data_process_${seed_num}.log
data_end=$(date +%s)
data_spend=$(( data_end - data_start ))
echo Dataset process time is ${data_spend} seconds.
