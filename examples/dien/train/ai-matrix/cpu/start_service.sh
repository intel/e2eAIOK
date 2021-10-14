source /etc/profile.d/spark-env.sh
/etc/init.d/ssh start
$HADOOP_HOME/sbin/start-dfs.sh
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-worker.sh spark://`hostname`:7077
