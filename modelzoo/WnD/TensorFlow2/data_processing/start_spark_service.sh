master_hostname=`hostname`
${SPARK_HOME}/sbin/start-master.sh
${SPARK_HOME}/sbin/start-worker.sh spark://${master_hostname}:7077
${SPARK_HOME}/sbin/start-history-server.sh