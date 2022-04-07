#!/bin/bash

# source spark env
source /etc/profile.d/spark-env.sh
# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# set vars
MODEL_NAME=wnd
DATA_PATH=/home/vmagent/app/dataset/outbrain

# create ci log dir
log_dir=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
mkdir -p /home/vmagent/app/cicd_logs/aidk_cicd_wnd_$log_dir

# clean ci temp files (pre stage)
rm /home/vmagent/app/hydro.ai/hydroai.db >/dev/null 2>&1
rm -rf /home/vmagent/app/hydro.ai/result >/dev/null 2>&1

set -e
# lauch AIDK wnd
cd /home/vmagent/app/hydro.ai
printf "y\ny\n" | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf conf/hydroai_defaults_wnd_example.conf 2>&1 | tee /home/vmagent/app/cicd_logs/aidk_cicd_wnd_$log_dir/aidk_cicd.log

# store aidk ci logs and data
cp /home/vmagent/app/hydro.ai/hydroai.db /home/vmagent/app/cicd_logs/aidk_cicd_wnd_$log_dir/
cp -r /home/vmagent/app/hydro.ai/result /home/vmagent/app/cicd_logs/aidk_cicd_wnd_$log_dir/

# check wnd
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_model_reload.bats

# clean ci temp files (post stage)
rm /home/vmagent/app/hydro.ai/hydroai.db >/dev/null 2>&1
rm -rf /home/vmagent/app/hydro.ai/result >/dev/null 2>&1