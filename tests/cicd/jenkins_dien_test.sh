#!/bin/bash

# source spark env
source /etc/profile.d/spark-env.sh
# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# set vars
MODEL_NAME=dien
DATA_PATH=/home/vmagent/app/dataset/amazon_reviews

# create ci log dir
log_dir=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
mkdir -p /home/vmagent/app/cicd_logs/aidk_cicd_dien_$log_dir

# set AIDK CI default options
USE_SIGOPT="${USE_SIGOPT:=1}"
set -e
# lauch AIDK dien
cd /home/vmagent/app/hydro.ai
if [ $USE_SIGOPT == 1 ]; then
  printf "y\ny\n" | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME | tee /home/vmagent/app/cicd_logs/aidk_cicd_dien_$log_dir/aidk_cicd.log
else
  printf "y\ny\n" | python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --no_sigopt | tee /home/vmagent/app/cicd_logs/aidk_cicd_dien_$log_dir/aidk_cicd.log
fi

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_result_exist.bats
#LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_model_reload.bats

cp /home/vmagent/app/hydro.ai/hydroai.db /home/vmagent/app/cicd_logs/aidk_cicd_dien_$log_dir/
cp -r /home/vmagent/app/hydro.ai/result /home/vmagent/app/cicd_logs/aidk_cicd_dien_$log_dir/
