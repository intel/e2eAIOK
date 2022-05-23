#!/bin/bash

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# set vars
MODEL_NAME="dien"
DATA_PATH="/home/vmagent/app/dataset/amazon_reviews"

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/home/vmagent/app/cicd_logs/aidk_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# set AIDK CI default options
USE_SIGOPT="${USE_SIGOPT:=1}"
set -e
# lauch AIDK dien
cd /home/vmagent/app/hydro.ai
if [ $USE_SIGOPT == 1 ]; then
  printf "y\ny\n" | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME | tee $tmp_dir/aidk_cicd.log
else
  printf "y\ny\n" | python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --no_sigopt | tee $tmp_dir/aidk_cicd.log
fi

# store aidk ci logs and data
cp /home/vmagent/app/hydro.ai/hydroai.db $tmp_dir/
cp -r /home/vmagent/app/hydro.ai/result $tmp_dir/

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_model_reload.bats
