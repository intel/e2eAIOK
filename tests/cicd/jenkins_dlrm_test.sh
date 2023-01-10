#!/bin/bash

# set vars
MODEL_NAME="dlrm"
DATA_PATH="/home/vmagent/app/dataset/criteo"
CONF_FILE="/home/vmagent/app/e2eaiok/tests/cicd/conf/e2eaiok_defaults_dlrm_example.conf"

# init conda
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/home/vmagent/app/cicd_logs/e2eaiok_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# source dlrm env vars
source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/

set -e
# lauch e2eAIOK dlrm
cd /home/vmagent/app/e2eaiok
if [ $USE_SIGOPT == 1 ]; then
  SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE --enable_sigopt 2>&1 | tee $tmp_dir/e2eaiok_cicd.log
else
  python run_e2eaiok.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE  2>&1 | tee $tmp_dir/e2eaiok_cicd.log
fi

# store e2eaiok ci logs and data
cp /home/vmagent/app/e2eaiok/e2eaiok.db $tmp_dir/
cp -r /home/vmagent/app/e2eaiok/result $tmp_dir/

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME LOG_FILE=$tmp_dir/e2eaiok_cicd.log tests/cicd/bats/bin/bats tests/cicd/test_log_format.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_model_reload.bats
