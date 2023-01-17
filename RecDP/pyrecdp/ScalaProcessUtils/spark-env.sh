export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export SPARK_HOME=${PYRECDP_HOME}/spark-3.2.1-bin-hadoop3.2
export PATH=$PATH:$SPARK_HOME/bin:$JAVA_HOME/jre/bin
export PYSPARK_PYTHON=`which python`
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
export PYSPARK_WORKER_PYTHON=$PYSPARK_PYTHON
# Add the PySpark classes to the Python path:
export PYTHONPATH=${SPARK_HOME}/python/:$PYTHONPATH
export PYTHONPATH=`find ${SPARK_HOME} -name py4j*zip`:$PYTHONPATH