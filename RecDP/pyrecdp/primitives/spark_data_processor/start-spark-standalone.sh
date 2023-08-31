if [ "$#" -ne 1 ]; then
    echo "Please provide master node, ex: " $0 "127.0.0.1"
fi
master_log=spark_master.log
nohup -- spark-class org.apache.spark.deploy.master.Master -h 0.0.0.0 >> ${master_log} 2>&1 < /dev/null &
master_pid="$!"

worker_log=spark_worker.log
nohup -- spark-class org.apache.spark.deploy.worker.Worker spark://${1}:7077 >> ${worker_log} 2>&1 < /dev/null &
worker_pid="$!"

echo ${master_pid} > spark_master_pid
echo ${worker_pid} > spark_master_pid