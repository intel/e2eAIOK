if [ -z $SPARK_HOME ]; then
    source /root/spark-env.sh
fi
jupyter notebook --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --notebook-dir /home/vmagent/app/recdp/examples/notebooks
