#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export SPARK_HOME=/home/spark-3.2.1-bin-hadoop3.2
export HADOOP_CONF_DIR=$SPARK_HOME/conf/
export PYSPARK_PYTHON=`which python3`
export PATH=$PATH:$SPARK_HOME/bin
export PATH=$PATH:$JAVA_HOME/jre/bin
source ${SPARK_HOME}/bin/load-spark-env.sh
if [[ -z "$PYSPARK_DRIVER_PYTHON" ]]; then
  PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
fi
if [[ -z "$PYSPARK_WORKER_PYTHON" ]]; then
  PYSPARK_WORKER_PYTHON=$PYSPARK_PYTHON
fi

export HDFS_NAMENODE_USER="root"
export HDFS_DATANODE_USER="root"
export HDFS_SECONDARYNAMENODE_USER="root"
#[ -z `pgrep sshd` ] && /usr/sbin/sshd

# Add the PySpark classes to the Python path:
export PYTHONPATH=${SPARK_HOME}/python/:$PYTHONPATH
export PYTHONPATH=`find ${SPARK_HOME} -name py4j*zip`:$PYTHONPATH

