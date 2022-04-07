#!/bin/bash

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# set AIDK CI default options
# enabled: USE_SIGOPT, CODE_STYLE_CHECK
# disabled: CODE_STYLE_FORMAT, LOG_FORMAT_CHECK
USE_SIGOPT="${USE_SIGOPT:=1}"
CODE_STYLE_FORMAT="${CODE_STYLE_FORMAT:=0}"
CODE_STYLE_CHECK="${CODE_STYLE_CHECK:=1}"
LOG_FORMAT_CHECK="${LOG_FORMAT_CHECK:=0}"
# set vars
MODEL_NAME=pipeline_test
DATA_PATH=/home/vmagent/app/dataset/pipeline_test

# create ci log dir
log_dir=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
mkdir -p /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir

# static codestyle format
if [ $CODE_STYLE_FORMAT == 1 ]
then
  echo "================================================================================"
  echo "======================== AIDK codestyle format (YAPF) =========================="
  /opt/intel/oneapi/intelpython/latest/bin/yapf -i -r /home/vmagent/app/hydro.ai/SDA/
  /opt/intel/oneapi/intelpython/latest/bin/yapf -i -r /home/vmagent/app/hydro.ai/hydroai/
  echo "================================================================================"
fi
# static codestyle check
if [ $CODE_STYLE_CHECK == 1 ]
then
  echo "================================================================================"
  echo "======================= AIDK codestyle check (pylint) =========================="
  /opt/intel/oneapi/intelpython/latest/bin/pylint --output-format=json /home/vmagent/app/hydro.ai/SDA/ | /opt/intel/oneapi/intelpython/latest/bin/pylint-json2html -o /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir/pylint_out_SDA.html
  /opt/intel/oneapi/intelpython/latest/bin/pylint --output-format=json /home/vmagent/app/hydro.ai/hydroai/ | /opt/intel/oneapi/intelpython/latest/bin/pylint-json2html -o /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir/pylint_out_hydroai.html
  echo "================================================================================"
fi

# clean ci temp files (pre stage)
rm /home/vmagent/app/hydro.ai/hydroai.db >/dev/null 2>&1
rm -rf /home/vmagent/app/hydro.ai/result >/dev/null 2>&1

set -e
# lauch AIDK pipeline_test
cd /home/vmagent/app/hydro.ai
if [ $USE_SIGOPT == 1 ]; then
  yes | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN /opt/intel/oneapi/intelpython/latest/bin/python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH 2>&1 | tee /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir/aidk_cicd.log
else
  yes | /opt/intel/oneapi/intelpython/latest/bin/python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH --no_sigopt 2>&1 | tee /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir/aidk_cicd.log
fi

# store aidk ci logs and data
cp /home/vmagent/app/hydro.ai/hydroai.db /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir/
cp -r /home/vmagent/app/hydro.ai/result /home/vmagent/app/cicd_logs/aidk_cicd_pipeline_$log_dir/

# BATS UT for log format check
if [ $LOG_FORMAT_CHECK == 1 ]; then
  LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_log_format.bats
fi
# BATS UT for built-in workflow check: hydroai.db existence, model reload capability, model_terminate_context_resume capability
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_model_reload.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH /home/vmagent/app/hydro.ai/tests/cicd/bats/bin/bats /home/vmagent/app/hydro.ai/tests/cicd/test_terminate_resume.bats
# Pytest UT for built-in workflow check: model_saved_path, model accuracy, common utils
PYTHONPATH=/home/vmagent/app/hydro.ai /opt/intel/oneapi/intelpython/latest/bin/pytest /home/vmagent/app/hydro.ai/tests/cicd/src/test_pipeline_model_happypath.py
PYTHONPATH=/home/vmagent/app/hydro.ai /opt/intel/oneapi/intelpython/latest/bin/pytest /home/vmagent/app/hydro.ai/tests/cicd/src/test_pipeline_model_accuracy.py
PYTHONPATH=/home/vmagent/app/hydro.ai /opt/intel/oneapi/intelpython/latest/bin/pytest /home/vmagent/app/hydro.ai/tests/cicd/src/test_common_utils.py

# clean ci temp files (post stage)
rm /home/vmagent/app/hydro.ai/hydroai.db >/dev/null 2>&1
rm -rf /home/vmagent/app/hydro.ai/result >/dev/null 2>&1

# source spark env
source /etc/profile.d/spark-env.sh
export master_hostname=`hostname`
/home/spark-3.2.0-bin-hadoop3.2/sbin/start-master.sh
/home/spark-3.2.0-bin-hadoop3.2/sbin/start-worker.sh spark://${master_hostname}:7077
/opt/intel/oneapi/intelpython/latest/bin/python /home/vmagent/app/test/test_spark_local.py
/opt/intel/oneapi/intelpython/latest/bin/python /home/vmagent/app/hydro.ai/RecDP/tests/test_categorify.py

# UT check for docker env with horovod
echo "============================start check docker env=============================="
echo "================================================================================"
echo "============================ check docker horovod =============================="
/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/horovodrun -n 2 -H localhost:2 /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/pytest -v --capture=tee-sys /home/vmagent/app/hydro.ai/tests/cicd/src/test_docker_horovod.py
echo "================================================================================"
# UT check for docker env with torchccl
echo "============================ check docker torchccl ============================="
/opt/intel/oneapi/mpi/2021.4.0/bin/mpirun -n 2 -l /opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/bin/pytest -v --capture=tee-sys /home/vmagent/app/hydro.ai/tests/cicd/src/test_docker_torchccl.py
echo "================================================================================"
echo "============================finish check docker env============================="
