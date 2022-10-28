# source ~/.local/env/setvars.sh
set -ex
seed_num=$(date +%s)

start=$(date +%s)
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

# model training
echo "start model training"
train_start=$(date +%s)
cd ./dlrm
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ./launch.py --distributed --config-path="../data_processing/config.yaml" --save-path="../data_processing/data_info.txt" --ncpu_per_proc=1 --nproc_per_node=2 --nnodes=4 --world_size=8 --hostfile ../hosts --master_addr="10.112.228.4" $dlrm_extra_option 2>&1 | tee run_train_${seed_num}.log
train_end=$(date +%s)
train_spend=$(( train_end - train_start ))
echo training time is ${train_spend} seconds.

