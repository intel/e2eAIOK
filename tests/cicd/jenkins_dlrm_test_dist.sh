#!/bin/bash

# source oneapi env vars
source /root/.oneapi_env_vars
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
# set vars
MODEL_NAME=dlrm
DATA_PATH=/home/vmagent/app/dataset/criteo

# create ci log dir
log_dir=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
mkdir -p /home/vmagent/app/cicd_logs/aidk_cicd_dlrm_$log_dir

# source dlrm env vars
source /opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/

sed -i 's/12345/12346/g' /etc/ssh/sshd_config
sed -i 's/12345/12346/g' /etc/ssh/ssh_config
service ssh start
ssh-keyscan -p 12346 -H 10.1.2.208 >> /root/.ssh/known_hosts
ssh-keyscan -p 12346 -H 10.1.2.212 >> /root/.ssh/known_hosts

# set AIDK CI default options
USE_SIGOPT="${USE_SIGOPT:=1}"
set -e
# lauch AIDK dlrm
conda activate pytorch_mlperf
cd /home/vmagent/app/hydro.ai

if [ $USE_SIGOPT == 1 ]; then
  printf "y\ny\n" | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --no_sigopt --data_path $DATA_PATH --model_name $MODEL_NAME --conf tests/cicd/conf/hydroai_defaults_dlrm_dist_example.conf 2>&1 | tee /home/vmagent/app/cicd_logs/aidk_cicd_dlrm_$log_dir/aidk_cicd.log
else
  printf "y\ny\n" | python run_hydroai.py --no_sigopt --data_path $DATA_PATH --model_name $MODEL_NAME --conf tests/cicd/conf/hydroai_defaults_dlrm_dist_example.conf --no_sigopt 2>&1 | tee /home/vmagent/app/cicd_logs/aidk_cicd_dlrm_$log_dir/aidk_cicd.log
fi

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_result_exist.bats
#LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_model_reload.bats

cp /home/vmagent/app/hydro.ai/hydroai.db /home/vmagent/app/cicd_logs/aidk_cicd_dlrm_$log_dir/
cp -r /home/vmagent/app/hydro.ai/result /home/vmagent/app/cicd_logs/aidk_cicd_dlrm_$log_dir/
