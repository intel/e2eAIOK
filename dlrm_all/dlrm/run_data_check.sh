# bash run_data_check.sh criteo_small
# bash run_data_check.sh kaggle
# bash run_data_check.sh criteo_full
#!/bin/bash
set -e

pre_dir=$(pwd)
if [[ ${1} != "criteo_small" && ${1} != "criteo_full" && ${1} != "kaggle" ]]; then
    echo "error: need to use 'criteo_small' or 'criteo_full' or 'kaggle' mode"
    exit
fi

if [ "${1}" = "criteo_small" ]; then
    train_days="0-3"
fi

if [ "${1}" = "kaggle" ]; then
    train_days="24-24"
fi

if [ "${1}" = "criteo_full" ]; then
    train_days="0-22"
fi

data_path="/home/vmagent/app/dataset/criteo"
echo "check data path: $data_path"
if [ ! -d $data_path ];then
    echo "check failed, create data path"
    mkdir $data_path
fi
cd $data_path
if [ "${1}" = "kaggle" ]; then
    train_day="$data_path/train.txt"
    echo "check kaggle dataset"
    if [ ! -f $train_day ];then
        echo "check failed. download it"
        download_file="https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz"
        wget -c $download_file -O - | tar -xz
    fi
    line_info=$(wc -l train.txt)
    total_lines=${line_info% *}
    train_lines=$[ $total_lines*6/7 ]
    test_lines=$[ $total_lines-$train_lines ]
    cat train.txt | head -$train_lines > day_24
    cat train.txt | tail -$test_lines > day_25
    echo "kaggle dataset has been checked"
else
    download_path="https://storage.googleapis.com/criteo-cail-datasets"
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
fi

cd $pre_dir
echo "cd work_dir: $(pwd)"

