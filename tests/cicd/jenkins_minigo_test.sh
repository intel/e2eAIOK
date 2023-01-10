#!/bin/bash

# set vars
MODEL_NAME="minigo"
DATA_PATH="/mnt/DP_disk1/dataset/minigo"
CONF_FILE="tests/cicd/conf/e2eaiok_defaults_minigo_example.conf"

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --force

# init conda
eval "$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# create ci log dir
hashstr_id=$(date +%Y-%m-%d)_$(echo $RANDOM | md5sum | head -c 8)
tmp_dir="/root/cicd_logs/e2eaiok_cicd_"$MODEL_NAME"_"$hashstr_id
mkdir -p $tmp_dir

set -e
# lauch e2eaiok minigo
cd modelzoo/minigo
conda activate minigo
printf "\n" | ./cc/configure_tensorflow.sh
export HOME=/root
./ml_perf/scripts/cc_libgen_parallel_selfplay.sh
# make MiniGo CI/CD test process faster
sed -i '/--winrate=/ s/=.*/=0.003/' ml_perf/flags/19/train_loop.flags
sed -i '/--eval=/ s/=.*/=1/' ml_perf/flags/19/train_loop.flags
sed -i '/--num_games=/ s/=.*/=4096/' ml_perf/flags/19/bootstrap.flags
sed -i '/--min_games_per_iteration=/ s/=.*/=4096/' ml_perf/flags/19/train_loop.flags
cd ../../
[[ -d result ]] || mkdir result
if [ $USE_SIGOPT == 1 ]; then
  SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE --enable_sigopt --custom_result_path $tmp_dir 2>&1 | tee $tmp_dir/e2eaiok_cicd.log
else
  python run_e2eaiok.py --data_path $DATA_PATH --model_name $MODEL_NAME --conf $CONF_FILE  --custom_result_path $tmp_dir 2>&1 | tee $tmp_dir/e2eaiok_cicd.log
fi

# test
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_result_exist.bats
LANG=C SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN MODEL_NAME=$MODEL_NAME DATA_PATH=$DATA_PATH CUSTOM_RESULT_PATH=$tmp_dir tests/cicd/bats/bin/bats tests/cicd/test_model_reload.bats
