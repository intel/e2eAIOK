#!/usr/bin/env bash
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


echo 'Running with parameters:'
echo "    USE_CASE: ${USE_CASE}"
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    BACKBONE_MODEL_DIRECTORY: ${BACKBONE_MODEL_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo "    MOUNT_INTELAI_MODELS_COMMON_SOURCE_DIR: ${MOUNT_INTELAI_MODELS_COMMON_SOURCE}"
if [ -n "${DOCKER}" ]; then
  echo "    Mounted volumes:"
  echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
  echo "        ${EXTERNAL_MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_EXTERNAL_MODELS_SOURCE}"
  echo "        ${INTELAI_MODELS} mounted on: ${MOUNT_INTELAI_MODELS_SOURCE}"
  echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
  echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
  echo "        ${BACKBONE_MODEL_DIRECTORY_VOL} mounted on: ${BACKBONE_MODEL_DIRECTORY}"
fi

echo "    SOCKET_ID: ${SOCKET_ID}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PRECISION: ${PRECISION}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"
echo "    NUM_CORES: ${NUM_CORES}"
echo "    BENCHMARK_ONLY: ${BENCHMARK_ONLY}"
echo "    ACCURACY_ONLY: ${ACCURACY_ONLY}"
echo "    OUTPUT_RESULTS: ${OUTPUT_RESULTS}"
echo "    DISABLE_TCMALLOC: ${DISABLE_TCMALLOC}"
echo "    TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD: ${TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD}"
echo "    NOINSTALL: ${NOINSTALL}"
echo "    OUTPUT_DIR: ${OUTPUT_DIR}"
echo "    MPI_NUM_PROCESSES: ${MPI_NUM_PROCESSES}"
echo "    MPI_NUM_PEOCESSES_PER_SOCKET: ${MPI_NUM_PROCESSES_PER_SOCKET}"
echo "    MPI_HOSTNAMES: ${MPI_HOSTNAMES}"
echo "    NUMA_CORES_PER_INSTANCE: ${NUMA_CORES_PER_INSTANCE}"
echo "    PYTHON_EXE: ${PYTHON_EXE}"
echo "    PYTHONPATH: ${PYTHONPATH}"
echo "    DRY_RUN: ${DRY_RUN}"

#  inference & training is supported right now
if [ ${MODE} != "inference" ] && [ ${MODE} != "training" ]; then
  echo "${MODE} mode for ${MODEL_NAME} is not supported"
  exit 1
fi

# Determines if we are running in a container by checking for .dockerenv
function _running-in-container()
{
  # .dockerenv is a legacy mount populated by Docker engine and at some point it may go away.
  [ -f /.dockerenv ]
}

# Check the Linux platform distribution if CentOS or Ubuntu
OS_PLATFORM=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
OS_VERSION=$(awk -F= '/^VERSION_ID/{print $2}' /etc/os-release)
if [[ ${OS_PLATFORM} == *"CentOS"* ]]; then
  if [[ "${OS_VERSION}" != '"8"' ]]; then
    echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
    exit 1
  fi
elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]]; then
  if [[ "${OS_VERSION}" != '"18.04"' ]] && [[ "${OS_VERSION}" != '"20.04"' ]]; then
    echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
    exit 1
  fi
else
  echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
  exit 1
fi

echo "Running on ${OS_PLATFORM} version ${OS_VERSION} is supported."

if [[ ${NOINSTALL} != "True" ]]; then
  # set env var before installs so that user interaction is not required
  export DEBIAN_FRONTEND=noninteractive
  # install common dependencies
  if [[ ${OS_PLATFORM} == *"CentOS"* ]]; then
    #yum update -y
    #yum install -y gcc gcc-c++ cmake python3-tkinter libXext libSM

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      dnf -y install https://extras.getpagespeed.com/release-el8-latest.rpm && \
      dnf -y install gperftools && \
      yum clean all
    fi

    #if [[ ${MPI_NUM_PROCESSES} != "None" ]]; then
      ## Installing OpenMPI
      #yum install -y openmpi openmpi-devel openssh openssh-server
      #yum clean all
      #export PATH="/usr/lib64/openmpi/bin:${PATH}"

      ## Install Horovod
      #export HOROVOD_WITHOUT_PYTORCH=1
      #export HOROVOD_WITHOUT_MXNET=1
      #export HOROVOD_WITH_TENSORFLOW=1
      #export HOROVOD_VERSION=87094a4

      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      #yum install -y git make
      #yum clean all
      #python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
      # python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
    #fi
  elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]]; then
    apt-get update -y
    apt-get install gcc-8 g++-8 cmake python-tk -y
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
    apt-get install -y libsm6 libxext6 python3-dev

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      apt-get install google-perftools -y
    fi

    if [[ ${MPI_NUM_PROCESSES} != "None" ]]; then
      ## Installing OpenMPI
      apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev -y

      ## Install Horovod
      export HOROVOD_WITHOUT_PYTORCH=1
      export HOROVOD_WITHOUT_MXNET=1
      export HOROVOD_WITH_TENSORFLOW=1
      export HOROVOD_VERSION=87094a4

      apt-get update
      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      apt-get install -y --no-install-recommends --fix-missing cmake git
      python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
      # apt-get install -y --no-install-recommends --fix-missing cmake
      # python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
    fi
  fi
  python3 -m pip install --upgrade 'pip>=20.3.4'
  python3 -m pip install requests
fi

# Determine if numactl needs to be installed
INSTALL_NUMACTL="False"
if [[ $NUMA_CORES_PER_INSTANCE != "None" || $SOCKET_ID != "-1" || $NUM_CORES != "-1" ]]; then
  # The --numa-cores-per-instance, --socket-id, and --num-cores args use numactl
  INSTALL_NUMACTL="True"
elif [[ $MODEL_NAME == "bert_large" && $MODE == "training" && $MPI_NUM_PROCESSES != "None" ]]; then
  # BERT large training with MPI uses numactl
  INSTALL_NUMACTL="True"
elif [[ $MODEL_NAME == "wide_deep" ]]; then
  # TODO: Why Wide & Deep uses numactl always
  INSTALL_NUMACTL="True"
fi

# If we are running in a container, call the container_init.sh files
if _running-in-container ; then
  # For running inside a real CentOS container
  if [[ ${OS_PLATFORM} == *"CentOS"* ]]; then
    if [[ $INSTALL_NUMACTL == "True" ]] && [[ ${NOINSTALL} != "True" ]]; then
      yum update -y
      yum install -y numactl
    fi
  elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]]; then
    # For ubuntu, run the container_init.sh scripts
    if [ -f ${MOUNT_BENCHMARK}/common/${FRAMEWORK}/container_init.sh ]; then
      # Call the framework's container_init.sh, if it exists and we are running on ubuntu
      INSTALL_NUMACTL=$INSTALL_NUMACTL bash ${MOUNT_BENCHMARK}/common/${FRAMEWORK}/container_init.sh
    fi
    # Call the model specific container_init.sh, if it exists
    if [ -f ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/${PRECISION}/container_init.sh ]; then
      bash ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/${PRECISION}/container_init.sh
    fi
  fi
fi

verbose_arg=""
if [ ${VERBOSE} == "True" ]; then
  verbose_arg="--verbose"
fi

weight_sharing_arg=""
if [ ${WEIGHT_SHARING} == "True" ]; then
  weight_sharing_arg="--weight-sharing"
fi
accuracy_only_arg=""
if [ ${ACCURACY_ONLY} == "True" ]; then
  accuracy_only_arg="--accuracy-only"
fi

benchmark_only_arg=""
if [ ${BENCHMARK_ONLY} == "True" ]; then
  benchmark_only_arg="--benchmark-only"
fi

output_results_arg=""
if [ ${OUTPUT_RESULTS} == "True" ]; then
  output_results_arg="--output-results"
fi

numa_cores_per_instance_arg=""
if [[ -n ${NUMA_CORES_PER_INSTANCE} && ${NUMA_CORES_PER_INSTANCE} != "None" ]]; then
  numa_cores_per_instance_arg="--numa-cores-per-instance=${NUMA_CORES_PER_INSTANCE}"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

timestamp=`date +%Y%m%d_%H%M%S`
LOG_FILENAME="benchmark_${MODEL_NAME}_${MODE}_${PRECISION}_${timestamp}.log"
if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir ${OUTPUT_DIR}
fi

export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_COMMON_SOURCE}:${MOUNT_INTELAI_MODELS_SOURCE}

# Common execution command used by all models
function run_model() {
  # Navigate to the main benchmark directory before executing the script,
  # since the scripts use the benchmark/common scripts as well.
  cd ${MOUNT_BENCHMARK}

  # Start benchmarking
  if [[ -z $DRY_RUN ]]; then
    if [[ -z $numa_cores_per_instance_arg ]]; then
      eval ${CMD} 2>&1 | tee ${LOGFILE}
    else
      # Don't tee to a log file for numactl multi-instance runs
      eval ${CMD}
    fi
  else
    echo ${CMD}
    return
  fi

  if [ ${VERBOSE} == "True" ]; then
    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${LOGFILE}
    if [[ ${BATCH_SIZE} != "-1" ]]; then
      echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOGFILE}
    fi
  fi

  if [[ ${BATCH_SIZE} != "-1" ]]; then
    echo "Ran ${MODE} with batch size ${BATCH_SIZE}" | tee -a ${LOGFILE}
  fi

  # if it starts with /workspace then it's not a separate mounted dir
  # so it's custom and is in same spot as LOGFILE is, otherwise it's mounted in a different place
  if [[ "${OUTPUT_DIR}" = "/workspace"* ]]; then
    LOG_LOCATION_OUTSIDE_CONTAINER=${BENCHMARK_SCRIPTS}/common/${FRAMEWORK}/logs/${LOG_FILENAME}
  else
    LOG_LOCATION_OUTSIDE_CONTAINER=${LOGFILE}
  fi

  # Don't print log file location for numactl multi-instance runs, because those have
  # separate log files for each instance
  if [[ -z $numa_cores_per_instance_arg ]]; then
    echo "Log file location: ${LOG_LOCATION_OUTSIDE_CONTAINER}" | tee -a ${LOGFILE}
  fi
}

# basic run command with commonly used args
CMD="${PYTHON_EXE} ${RUN_SCRIPT_PATH} \
--framework=${FRAMEWORK} \
--use-case=${USE_CASE} \
--model-name=${MODEL_NAME} \
--precision=${PRECISION} \
--mode=${MODE} \
--benchmark-dir=${MOUNT_BENCHMARK} \
--intelai-models=${MOUNT_INTELAI_MODELS_SOURCE} \
--num-cores=${NUM_CORES} \
--batch-size=${BATCH_SIZE} \
--socket-id=${SOCKET_ID} \
--output-dir=${OUTPUT_DIR} \
--num-train-steps=${NUM_TRAIN_STEPS} \
${numa_cores_per_instance_arg} \
${accuracy_only_arg} \
${benchmark_only_arg} \
${output_results_arg} \
${weight_sharing_arg} \
${verbose_arg}"

if [ ${MOUNT_EXTERNAL_MODELS_SOURCE} != "None" ]; then
  CMD="${CMD} --model-source-dir=${MOUNT_EXTERNAL_MODELS_SOURCE}"
fi

if [[ -n "${IN_GRAPH}" && ${IN_GRAPH} != "" ]]; then
  CMD="${CMD} --in-graph=${IN_GRAPH}"
fi

if [[ -n "${CHECKPOINT_DIRECTORY}" && ${CHECKPOINT_DIRECTORY} != "" ]]; then
  CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY}"
fi

if [[ -n "${BACKBONE_MODEL_DIRECTORY}" && ${BACKBONE_MODEL_DIRECTORY} != "" ]]; then
  CMD="${CMD} --backbone-model=${BACKBONE_MODEL_DIRECTORY}"
fi

if [[ -n "${DATASET_LOCATION}" && ${DATASET_LOCATION} != "" ]]; then
  CMD="${CMD} --data-location=${DATASET_LOCATION}"
fi

if [ ${NUM_INTER_THREADS} != "None" ]; then
  CMD="${CMD} --num-inter-threads=${NUM_INTER_THREADS}"
fi

if [ ${NUM_INTRA_THREADS} != "None" ]; then
  CMD="${CMD} --num-intra-threads=${NUM_INTRA_THREADS}"
fi

if [ ${DATA_NUM_INTER_THREADS} != "None" ]; then
  CMD="${CMD} --data-num-inter-threads=${DATA_NUM_INTER_THREADS}"
fi

if [ ${DATA_NUM_INTRA_THREADS} != "None" ]; then
  CMD="${CMD} --data-num-intra-threads=${DATA_NUM_INTRA_THREADS}"
fi

if [ ${DISABLE_TCMALLOC} != "None" ]; then
  CMD="${CMD} --disable-tcmalloc=${DISABLE_TCMALLOC}"
fi

## Added for bert
function bert_options() {

  if [[ ${MODE} == "training" ]]; then
    if [[ -z "${train_option}" ]]; then
      echo "Error: Please specify a train option (SQuAD, Classifier, Pretraining)"
      exit 1
    fi

    CMD=" ${CMD} --train-option=${train_option}"
  fi

  if [[ ${MODE} == "inference" ]]; then
    if [[ -z "${infer_option}" ]]; then
      echo "Error: Please specify a inference option (SQuAD, Classifier, Pretraining)"
      exit 1
    fi

    CMD=" ${CMD} --infer-option=${infer_option}"
  fi

  if [[ -n "${init_checkpoint}" && ${init_checkpoint} != "" ]]; then
    CMD=" ${CMD} --init-checkpoint=${init_checkpoint}"
  fi

  if [[ -n "${task_name}" && ${task_name} != "" ]]; then
    CMD=" ${CMD} --task-name=${task_name}"
  fi

  if [[ -n "${warmup_steps}" && ${warmup_steps} != "" ]]; then
    CMD=" ${CMD} --warmup-steps=${warmup_steps}"
  fi

  if [[ -n "${steps}" && ${steps} != "" ]]; then
    CMD=" ${CMD} --steps=${steps}"
  fi

  if [[ -n "${vocab_file}" && ${vocab_file} != "" ]]; then
    CMD=" ${CMD} --vocab-file=${vocab_file}"
  fi

  if [[ -n "${config_file}" && ${config_file} != "" ]]; then
    CMD=" ${CMD} --config-file=${config_file}"
  fi

  if [[ -n "${do_predict}" && ${do_predict} != "" ]]; then
    CMD=" ${CMD} --do-predict=${do_predict}"
  fi

  if [[ -n "${predict_file}" && ${predict_file} != "" ]]; then
    CMD=" ${CMD} --predict-file=${predict_file}"
  fi

  if [[ -n "${do_train}" && ${do_train} != "" ]]; then
    CMD=" ${CMD} --do-train=${do_train}"
  fi

  if [[ -n "${train_file}" && ${train_file} != "" ]]; then
    CMD=" ${CMD} --train-file=${train_file}"
  fi

  if [[ -n "${num_train_epochs}" && ${num_train_epochs} != "" ]]; then
    CMD=" ${CMD} --num-train-epochs=${num_train_epochs}"
  fi

  if [[ -n "${num_train_steps}" && ${num_train_steps} != "" ]]; then
    CMD=" ${CMD} --num-train-steps=${num_train_steps}"
  fi

  if [[ -n "${max_predictions}" && ${max_predictions} != "" ]]; then
    CMD=" ${CMD} --max-predictions=${max_predictions}"
  fi

  if [[ -n "${learning_rate}" && ${learning_rate} != "" ]]; then
    CMD=" ${CMD} --learning-rate=${learning_rate}"
  fi

  if [[ -n "${max_seq_length}" && ${max_seq_length} != "" ]]; then
    CMD=" ${CMD} --max-seq-length=${max_seq_length}"
  fi

  if [[ -n "${doc_stride}" && ${doc_stride} != "" ]]; then
    CMD=" ${CMD} --doc-stride=${doc_stride}"
  fi

  if [[ -n "${input_file}" && ${input_file} != "" ]]; then
    CMD=" ${CMD} --input-file=${input_file}"
  fi

  if [[ -n "${do_eval}" && ${do_eval} != "" ]]; then
    CMD=" ${CMD} --do-eval=${do_eval}"
  fi

  if [[ -n "${data_dir}" && ${data_dir} != "" ]]; then
    CMD=" ${CMD} --data-dir=${data_dir}"
  fi

  if [[ -n "${do_lower_case}" && ${do_lower_case} != "" ]]; then
    CMD=" ${CMD} --do-lower-case=${do_lower_case}"
  fi
  if [[ -n "${accum_steps}" && ${accum_steps} != "" ]]; then
    CMD=" ${CMD} --accum_steps=${accum_steps}"
  fi
  if [[ -n "${profile}" && ${profile} != "" ]]; then
    CMD=" ${CMD} --profile=${profile}"
  fi
  if [[ -n "${experimental_gelu}" && ${experimental_gelu} != "" ]]; then
    CMD=" ${CMD} --experimental-gelu=${experimental_gelu}"
  fi
  if [[ -n "${optimized_softmax}" && ${optimized_softmax} != "" ]]; then
    CMD=" ${CMD} --optimized-softmax=${optimized_softmax}"
  fi

  if [[ -n "${mpi_workers_sync_gradients}" && ${mpi_workers_sync_gradients} != "" ]]; then
    CMD=" ${CMD} --mpi_workers_sync_gradients=${mpi_workers_sync_gradients}"
  fi

  if [[ -n "${num_hidden_layers}" && ${num_hidden_layers} != "" ]]; then
    CMD=" ${CMD} --num_hidden_layers=${num_hidden_layers}"
  fi
  if [[ -n "${attention_probs_dropout_prob}" && ${attention_probs_dropout_prob} != "" ]]; then
    CMD=" ${CMD} --attention_probs_dropout_prob=${attention_probs_dropout_prob}"
  fi
  if [[ -n "${hidden_dropout_prob}" && ${hidden_dropout_prob} != "" ]]; then
    CMD=" ${CMD} --hidden_dropout_prob=${hidden_dropout_prob}"
  fi
  if [[ -n "${host_file}" && ${host_file} != "" ]]; then
    CMD=" ${CMD} --host_file=${host_file}"
  fi
  if [[ -n "${test_file}" && ${test_file} != "" ]]; then
    CMD=" ${CMD} --test-file=${test_file}"
  fi
  if [[ -n "${num_to_evaluate}" && ${num_to_evaluate} != "" ]]; then
    CMD=" ${CMD} --num-to-evaluate=${num_to_evaluate}"
  fi
  if [[ -n "${f1_threshold}" && ${f1_threshold} != "" ]]; then
    CMD=" ${CMD} --f1-threshold=${f1_threshold}"
  fi

}

function install_protoc() {
  pushd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"

  # install protoc, if necessary, then compile protoc files
  if [ ! -f "bin/protoc" ]; then
    install_location=$1
    echo "protoc not found, installing protoc from ${install_location}"
    if [[ ${OS_PLATFORM} == *"CentOS"* ]]; then
      yum update -y && yum install -y unzip wget
    else
      apt-get update && apt-get install -y unzip wget
    fi
    wget -O protobuf.zip ${install_location}
    unzip -o protobuf.zip
    rm protobuf.zip
  else
    echo "protoc already found"
  fi

  echo "Compiling protoc files"
  ./bin/protoc object_detection/protos/*.proto --python_out=.
  popd
}

function get_cocoapi() {
  # get arg for where the cocoapi repo was cloned
  cocoapi_dir=${1}

  # get arg for the location where we want the pycocotools
  parent_dir=${2}
  pycocotools_dir=${parent_dir}/pycocotools

  # If pycoco tools aren't already found, then builds the coco python API
  if [ ! -d ${pycocotools_dir} ]; then
    # This requires that the cocoapi is cloned in the external model source dir
    if [ -d "${cocoapi_dir}/PythonAPI" ]; then
      # install cocoapi
      pushd ${cocoapi_dir}/PythonAPI
      echo "Installing COCO API"
      make
      cp -r pycocotools ${parent_dir}
      popd
    else
      echo "${cocoapi_dir}/PythonAPI directory was not found"
      echo "Unable to install the python cocoapi."
      exit 1
    fi
  else
    echo "pycocotools were found at: ${pycocotools_dir}"
  fi
}

function add_arg() {
  local arg_str=""
  if [ -n "${2}" ]; then
    arg_str=" ${1}=${2}"
  fi
  echo "${arg_str}"
}

function add_steps_args() {
  # returns string with --steps and --warmup_steps, if there are values specified
  local steps_arg=""
  local trainepochs_arg=""
  local epochsbtweval_arg=""
  local warmup_steps_arg=""
  local kmp_blocktime_arg=""

  if [ -n "${steps}" ]; then
    steps_arg="--steps=${steps}"
  fi

  if [ -n "${train_epochs}" ]; then
    trainepochs_arg="--train_epochs=${train_epochs}"
  fi

  if [ -n "${epochs_between_evals}" ]; then
    epochsbtweval_arg="--epochs_between_evals=${epochs_between_evals}"
  fi

  if [ -n "${warmup_steps}" ]; then
    warmup_steps_arg="--warmup-steps=${warmup_steps}"
  fi

  if [ -n "${kmp_blocktime}" ]; then
    kmp_blocktime_arg="--kmp-blocktime=${kmp_blocktime}"
  fi

  echo "${steps_arg} ${trainepochs_arg} ${epochsbtweval_arg} ${warmup_steps_arg} ${kmp_blocktime_arg}"
}

function add_calibration_arg() {
  # returns string with --calibration-only, if True is specified,
  # in this case a subset (~ 100 images) of the ImageNet dataset
  # is generated to be used later on in calibrating the Int8 model.
  # also this function returns a string with --calibrate, if True is specified,
  # which enables resnet50 Int8 benchmark to run accuracy using the previously
  # generated ImageNet data subset.
  local calibration_arg=""

  if [[ ${calibration_only} == "True" ]]; then
    calibration_arg="--calibration-only"
  elif [[ ${calibrate} == "True" ]]; then
    calibration_arg="--calibrate=True"
  fi

  echo "${calibration_arg}"
}

#BERT model
function bert() {
   if [ ${PRECISION} == "fp32" ]; then
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      apt-get update && apt-get install -y git
      python3 -m pip install -r ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt
    fi

    CMD="${CMD} \
    $(add_arg "--task_name" ${task_name}) \
    $(add_arg "--max_seq_length" ${max_seq_length}) \
    $(add_arg "--eval_batch_size" ${eval_batch_size}) \
    $(add_arg "--learning_rate" ${learning_rate}) \
    $(add_arg "--vocab_file" ${vocab_file}) \
    $(add_arg "--bert_config_file" ${bert_config_file}) \
    $(add_arg "--init_checkpoint" ${init_checkpoint})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer language model from official tensorflow models
function transformer_lt_official() {
  if [ ${PRECISION} == "fp32" ]; then

    if [[ -z "${file}" ]]; then
        echo "transformer-language requires -- file arg to be defined"
        exit 1
    fi
    if [[ -z "${file_out}" ]]; then
        echo "transformer-language requires -- file_out arg to be defined"
        exit 1
    fi
    if [[ -z "${reference}" ]]; then
        echo "transformer-language requires -- reference arg to be defined"
        exit 1
    fi
    if [[ -z "${vocab_file}" ]]; then
        echo "transformer-language requires -- vocab_file arg to be defined"
        exit 1
    fi

    if [ ${NOINSTALL} != "True" ]; then
      python3 -m pip install -r "${MOUNT_BENCHMARK}/language_translation/tensorflow/transformer_lt_official/requirements.txt"
    fi

    CMD="${CMD}
    --vocab_file=${DATASET_LOCATION}/${vocab_file} \
    --file=${DATASET_LOCATION}/${file} \
    --file_out=${OUTPUT_DIR}/${file_out} \
    --reference=${DATASET_LOCATION}/${reference}"
    PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer in mlperf Translation for Tensorflow  model
function transformer_mlperf() {
  export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}
  if [[ ${MODE} == "training" ]]; then
    #pip install tensorflow-addons==0.6.0  #/workspace/benchmarks/common/tensorflow/tensorflow_addons-0.6.0.dev0-cp36-cp36m-linux_x86_64.whl
    if [[ (${PRECISION} == "bfloat16") || ( ${PRECISION} == "fp32") ]]
    then

      if [[ -z "${random_seed}" ]]; then
          echo "transformer-language requires --random_seed arg to be defined"
          exit 1
      fi
      if [[ -z "${params}" ]]; then
          echo "transformer-language requires --params arg to be defined"
          exit 1
      fi
      if [[ -z "${train_steps}" ]]; then
          echo "transformer-language requires --train_steps arg to be defined"
          exit 1
      fi
      if [[ -z "${steps_between_eval}" ]]; then
          echo "transformer-language requires --steps_between_eval arg to be defined"
          exit 1
      fi
      if [[ -z "${do_eval}" ]]; then
          echo "transformer-language requires --do_eval arg to be defined"
          exit 1
      fi
      if [[ -z "${save_checkpoints}" ]]; then
          echo "transformer-language requires --save_checkpoints arg to be defined"
          exit 1
      fi
      if [[ -z "${print_iter}" ]]; then
          echo "transformer-language requires --print_iter arg to be defined"
          exit 1
      fi

      CMD="${CMD} --random_seed=${random_seed} --params=${params} --train_steps=${train_steps} --steps_between_eval=${steps_between_eval} --do_eval=${do_eval} --save_checkpoints=${save_checkpoints}
      --print_iter=${print_iter} --save_profile=${save_profile}"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
  fi

  if [[ ${MODE} == "inference" ]]; then
    if [[ (${PRECISION} == "bfloat16") || ( ${PRECISION} == "fp32") || ( ${PRECISION} == "int8") ]]; then

      if [[ -z "${params}" ]]; then
          echo "transformer-language requires --params arg to be defined"
          exit 1
      fi

      if [[ -z "${file}" ]]; then
          echo "transformer-language requires -- file arg to be defined"
          exit 1
      fi
      if [[ -z "${file_out}" ]]; then
          echo "transformer-language requires -- file_out arg to be defined"
          exit 1
      fi
      if [[ -z "${reference}" ]]; then
          echo "transformer-language requires -- reference arg to be defined"
          exit 1
      fi

      CMD="${CMD} $(add_steps_args) $(add_arg "--params" ${params}) \
           $(add_arg "--file" ${DATASET_LOCATION}/${file}) \
           $(add_arg "--vocab_file" ${DATASET_LOCATION}/${vocab_file}) \
           $(add_arg "--file_out" ${OUTPUT_DIR}/${file_out}) \
           $(add_arg "--reference" ${DATASET_LOCATION}/${reference})"
      echo $CMD

      PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model

    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
  fi
}

# BERT base
function bert_base() {
  if [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "bfloat16" ]; then
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    bert_options
    CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# BERT Large model
function bert_large() {
    # Change if to support fp32
    if [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "int8" ] || [ $PRECISION == "bfloat16" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      bert_options
      #echo ${CMD}
      #exit 1
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
}


LOGFILE=${OUTPUT_DIR}/${LOG_FILENAME}

MODEL_NAME=$(echo ${MODEL_NAME} | tr 'A-Z' 'a-z')
if [ ${MODEL_NAME} == "bert" ]; then
  bert
elif [ ${MODEL_NAME} == "transformer_lt_official" ]; then
  transformer_lt_official
elif [ ${MODEL_NAME} == "transformer_mlperf" ]; then
  transformer_mlperf
elif [ ${MODEL_NAME} == "bert_base" ]; then
  bert_base
elif [ ${MODEL_NAME} == "bert_large" ]; then
  bert_large
else
  echo "Unsupported model: ${MODEL_NAME}"
  exit 1
fi
