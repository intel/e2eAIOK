#!/bin/bash

# set vars
MODEL_NAME="pipeline_test"
DATA_PATH="/home/vmagent/app/dataset/pipeline_test"

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/home/vmagent/app/cicd_logs/e2eaiok_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# set default options
CODE_STYLE_FORMAT="${CODE_STYLE_FORMAT:=0}"
CODE_STYLE_CHECK="${CODE_STYLE_CHECK:=0}"
# static codestyle format
if [ $CODE_STYLE_FORMAT == 1 ]
then
  yapf -i -r /home/vmagent/app/e2eaiok/SDA/
  yapf -i -r /home/vmagent/app/e2eaiok/e2eaiok/
fi
# static codestyle check
if [ $CODE_STYLE_CHECK == 1 ]
then
  pylint --output-format=json /home/vmagent/app/e2eaiok/SDA/ | pylint-json2html -o $tmp_dir/pylint_out_SDA.html
  pylint --output-format=json /home/vmagent/app/e2eaiok/e2eaiok/ | pylint-json2html -o $tmp_dir/pylint_out_e2eaiok.html
fi

set -e
# lauch e2eaiok pipeline_test
cd /home/vmagent/app/e2eaiok
if [ $USE_SIGOPT == 1 ]; then
  SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --model_name $MODEL_NAME --data_path $DATA_PATH --custom_result_path $tmp_dir 2>&1 | tee $tmp_dir/e2eaiok_cicd.log
else
  python run_e2eaiok.py --model_name $MODEL_NAME --data_path $DATA_PATH --no_sigopt --custom_result_path $tmp_dir 2>&1 | tee $tmp_dir/e2eaiok_cicd.log
fi

# BATS test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_model_reload.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_terminate_resume.bats
# Pytest test
PYTHONPATH=/home/vmagent/app/e2eaiok CUSTOM_RESULT_PATH=$tmp_dir pytest tests/cicd/src/test_pipeline_model_happypath.py
PYTHONPATH=/home/vmagent/app/e2eaiok CUSTOM_RESULT_PATH=$tmp_dir pytest tests/cicd/src/test_pipeline_model_accuracy.py
PYTHONPATH=/home/vmagent/app/e2eaiok pytest tests/cicd/src/test_common_utils.py
# Spark test
export master_hostname=`hostname`
/home/spark-3.2.1-bin-hadoop3.2/sbin/start-master.sh
/home/spark-3.2.1-bin-hadoop3.2/sbin/start-worker.sh spark://${master_hostname}:7077
python tests/cicd/src/test_spark_local.py --hostname ${master_hostname} 
python tests/cicd/src/test_categorify.py
