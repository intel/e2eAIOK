#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# set vars
MODEL_NAME="minigo"
DATA_PATH="/mnt/DP_disk1/dataset/minigo"
CONF_FILE="conf/hydroai_defaults_minigo_example.conf"

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/root/cicd_logs/aidk_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

# set AIDK CI default options
USE_SIGOPT="${USE_SIGOPT:=1}"
set -e
# lauch AIDK minigo
cd modelzoo/third_party/mlperf_v1.0/Intel/benchmarks/minigo/8-nodes-64s-8376H-tensorflow
conda activate minigo_xeon_opt
yes "" | ./cc/configure_tensorflow.sh
sed -i '/--winrate=/ s/=.*/=0.003/' ml_perf/flags/19/train_loop.flags
cd ../../../../../../../
[[ -d result ]] || mkdir result
if [ $USE_SIGOPT == 1 ]; then
  printf "y\ny\n" | SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE 2>&1 | tee $tmp_dir/aidk_cicd.log
else
  printf "y\ny\n" | python run_hydroai.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE --no_sigopt 2>&1 | tee $tmp_dir/aidk_cicd.log
fi

# store aidk ci logs and data
cp hydroai.db $tmp_dir/
cp -r result $tmp_dir/
