source /etc/profile.d/spark-env.sh
cd /home/vmagent/app/e2eaiok/
python setup.py install
pip install pyrecdp
sh /home/vmagent/app/e2eaiok/conf/spark/start_spark_service.sh 
python modelzoo/dien/feature_engineering/preprocessing.py --train
python modelzoo/dien/feature_engineering/preprocessing.py --test
source /opt/intel/oneapi/setvars.sh --force
conda activate tensorflow
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=4
export HOROVOD_CPU_OPERATIONS=CCL
# export MKLDNN_VERBOSE=2
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY="0,32"
export HOROVOD_THREAD_AFFINITY="1,33"
#export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="0,1,32,33"
python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien 