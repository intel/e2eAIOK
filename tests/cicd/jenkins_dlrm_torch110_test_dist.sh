#!/bin/bash

# source spark env
source /etc/profile.d/spark-env.sh
# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# set vars
MODEL_NAME="dlrm_torch110"
DATA_PATH="/home/vmagent/app/dataset/criteo"
CONF_FILE="/home/vmagent/app/hydro.ai/tests/cicd/conf/hydroai_defaults_dlrm_dist_example.conf"

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/home/vmagent/app/cicd_logs/aidk_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# config passwordless ssh
service ssh start
ssh-keyscan -p 12345 -H 10.1.2.212 >> /root/.ssh/known_hosts
ssh-keyscan -p 12345 -H 10.1.2.213 >> /root/.ssh/known_hosts

# set AIDK CI default options
USE_SIGOPT="${USE_SIGOPT:=1}"
set -e
# lauch AIDK dlrm torch110
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
