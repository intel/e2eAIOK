# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#########################################################################
# File Name: run-spark.sh

#!/bin/bash

set -e

# the environment variables to run spark job
# should modify below environment variables

# the data path including 1TB criteo data, day_0, day_1, ...
# HDFS folder 
export INPUT_PATH=${1:-'/kaggle/input'}
echo $INPUT_PATH

# the output path, use for generating the dictionary and the final dataset
# the output folder should have more than 300GB
export OUTPUT_PATH=${2:-'/kaggle/output'}

# spark local dir should have about 3TB
# the temporary path used for spark shuffle write
# Jian: change to NVMe drive
export SPARK_LOCAL_DIRS='/mnt/DP_disk4'

# below numbers should be adjusted according to the resource of your running environment
# set the total number of CPU cores, spark can use
export TOTAL_CORES=240

# set the number of executors
export NUM_EXECUTORS=12

# the cores for each executor, it'll be calculated
export NUM_EXECUTOR_CORES=$((${TOTAL_CORES}/${NUM_EXECUTORS}))

# unit: GB,  set the max memory you want to use
export TOTAL_MEMORY=1500

# unit: GB, set the memory for driver
export DRIVER_MEMORY=32

# the memory per executor
export EXECUTOR_MEMORY=$(((${TOTAL_MEMORY}-${DRIVER_MEMORY})/${NUM_EXECUTORS}))

# use frequency_limit=15 or not
# by default use a frequency limit of 15
USE_FREQUENCY_LIMIT=1
OPTS=""
if [[ $USE_FREQUENCY_LIMIT == 1 ]]; then
    OPTS="--frequency_limit 15"
fi

export SPARK_HOME=/home/xianyang/sw/spark-3.0.0-preview2-bin-hadoop2.7
export JAVA_HOME=/usr/lib/jvm/java-1.8.0
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# we use spark standalone to run the job
export MASTER=spark://sr231:7077

echo "Starting spark standalone"
#start-master.sh
#start-slave.sh $MASTER
echo "Generating the dictionary..."
spark-submit --master $MASTER \
        --driver-memory "${DRIVER_MEMORY}G" \
        --executor-cores $NUM_EXECUTOR_CORES \
        --executor-memory "${EXECUTOR_MEMORY}G" \
        --conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
        --conf spark.sql.shuffle.partitions=600 \
        --conf spark.driver.maxResultSize=2G \
        --conf spark.locality.wait=0s \
        --conf spark.network.timeout=1800s \
        spark_data_utils_kaggle.py --mode generate_models \
        $OPTS \
        --input_folder $INPUT_PATH \
        --raw_data_file "/kaggle/input/train.txt" \
        --days 0-6 \
        --model_folder $OUTPUT_PATH/models \
        --write_mode overwrite --low_mem 2>&1 | tee submit_dict_log.txt


echo "Transforming the train data from day_0 to day_6..."
spark-submit --master $MASTER \
        --driver-memory "${DRIVER_MEMORY}G" \
        --executor-cores $NUM_EXECUTOR_CORES \
        --executor-memory "${EXECUTOR_MEMORY}G" \
        --conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
        --conf spark.sql.shuffle.partitions=600 \
        --conf spark.driver.maxResultSize=2G \
        --conf spark.locality.wait=0s \
        --conf spark.network.timeout=1800s \
        spark_data_utils_kaggle.py --mode transform \
        --input_folder $INPUT_PATH \
        --raw_data_file "/kaggle/input/train.txt" \
        --days 0-6 \
        --output_folder $OUTPUT_PATH/train \
        --model_size_file $OUTPUT_PATH/model_size.json \
        --model_folder $OUTPUT_PATH/models \
        --write_mode overwrite --low_mem 2>&1 | tee submit_train_log.txt


