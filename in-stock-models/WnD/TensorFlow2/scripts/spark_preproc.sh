#!/bin/bash

export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip
export SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info"
export PYSPARK_PYTHON=/root/sw/miniconda3/envs/spark/bin/python
export PYSPARK_DRIVER_PYTHON=/root/sw/miniconda3/envs/spark/bin/python


tfrecords=${2:-40}

time /root/sw/miniconda3/envs/spark/bin/python -m data.outbrain.spark.preproc --num_train_partitions "${tfrecords}" --num_valid_partitions "${tfrecords}"