#!/bin/bash

# set vars
MODEL_NAME="dlrm"
DATA_PATH="/home/vmagent/app/dataset/criteo"
CONF_FILE="/home/vmagent/app/hydro.ai/tests/cicd/conf/hydroai_defaults_dlrm_example.conf"

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/home/vmagent/app/cicd_logs/aidk_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# source dlrm env vars
source /opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/

set -e
# lauch AIDK dlrm
cd /home/vmagent/app/hydro.ai
if [ $USE_SIGOPT == 1 ]; then
  SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE 2>&1 | tee $tmp_dir/aidk_cicd.log
else
  python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE --no_sigopt 2>&1 | tee $tmp_dir/aidk_cicd.log
fi

# store aidk ci logs and data
cp /home/vmagent/app/hydro.ai/hydroai.db $tmp_dir/
cp -r /home/vmagent/app/hydro.ai/result $tmp_dir/

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME LOG_FILE=$tmp_dir/aidk_cicd.log tests/cicd/bats/bin/bats tests/cicd/test_log_format.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_model_reload.bats
