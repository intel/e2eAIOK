#!/bin/bash

# source oneapi env vars
source /root/.oneapi_env_vars
# source spark env
source /etc/profile.d/spark-env.sh
# init conda env
__conda_setup="$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/intel/oneapi/intelpython/latest/etc/profile.d/conda.sh" ]; then
        . "/opt/intel/oneapi/intelpython/latest/etc/profile.d/conda.sh"
    else
        export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"
    fi
fi
unset __conda_setup
# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# set vars
MODEL_NAME="dlrm"
DATA_PATH="/home/vmagent/app/dataset/criteo"
CONF_FILE="/home/vmagent/app/hydro.ai/tests/cicd/conf/hydroai_defaults_dlrm_example.conf"

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/home/vmagent/app/cicd_logs/aidk_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# source dlrm env vars
source /opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/

# set AIDK CI default options
USE_SIGOPT="${USE_SIGOPT:=1}"
set -e
# lauch AIDK dlrm
cd /home/vmagent/app/hydro.ai
if [ $USE_SIGOPT == 1 ]; then
  printf "y\ny\n" | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE 2>&1 | tee $tmp_dir/aidk_cicd.log
else
  printf "y\ny\n" | python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE --no_sigopt 2>&1 | tee $tmp_dir/aidk_cicd.log
fi

# store aidk ci logs and data
cp /home/vmagent/app/hydro.ai/hydroai.db $tmp_dir/
cp -r /home/vmagent/app/hydro.ai/result $tmp_dir/

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_model_reload.bats
