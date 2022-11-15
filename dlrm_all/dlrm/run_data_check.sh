# bash run_train.sh local_small
# bash run_train.sh distributed_full
#!/bin/bash
set -e

pre_dir=$(pwd)
if [[ ${1} != "local_small" && ${1} != "distributed_full" ]]; then
    echo "error: need to use 'local_small' or 'distributed_full' mode"
    exit
fi

if [ "${1}" = "local_small" ]; then
    train_days="0-3"
fi
if [ "${1}" = "distributed_full" ]; then
    train_days="0-22"
fi

data_path="/home/vmagent/app/dataset/criteo"
download_path="https://storage.googleapis.com/criteo-cail-datasets"
echo "check data path: $data_path"
if [ ! -d $data_path ];then
    echo "check failed, create data path"
    mkdir $data_path
fi
cd $data_path
day_start=${train_days%-*}
day_end=${train_days#*-}

echo "check train dataset: day_$day_start ~ day_$day_end"
for ((day_i=$day_start; day_i<=$day_end; day_i++))
do
    train_day="$data_path/day_$day_i"
    echo "check $train_day"
    if [ ! -f $train_day ];then
        echo "day_$day_i does not exist. download it"
        download_file="$download_path/day_$day_i.gz"
        curl -O $download_file
        gzip -d "day_$day_i.gz"
    fi
done
echo "train dataset has been checked"

echo "check test dataset: day_23"
test_day="$data_path/day_23"
echo "check $test_day"
if [ ! -f $test_day ];then
    echo "day_23 does not exist. download it"
    download_file="$download_path/day_23.gz"
    curl -O $download_file
    gzip -d "day_23.gz"
fi
echo "test dataset has been checked"

cd $pre_dir
pwd

