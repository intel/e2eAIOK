#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export PYSPARK_PYTHON=`which python`
export SPARK_HOME=`${PYSPARK_PYTHON} -c "import pyspark; print(pyspark.__path__[0])"`
export HADOOP_CONF_DIR=$SPARK_HOME/conf/
export PATH=$PATH:$SPARK_HOME/bin
export PATH=$PATH:$JAVA_HOME/jre/bin
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