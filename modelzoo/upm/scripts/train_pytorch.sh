set -x
set -e
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
# CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)

 export CCL_WORKER_COUNT=2
 export CCL_WORKER_AFFINITY="16,17,34,35"
 export HOROVOD_THREAD_AFFINITY="53,71"
 export I_MPI_PIN_DOMAIN=socket
 export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="16,17,34,35,52,53,70,71"


time mpirun -genv OMP_NUM_THREADS=16 -map-by socket -n 2 -ppn 2 -hosts sr112 -print-rank-map \
-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 \
-iface eth3 \
/opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/bin/python -u main.py \
  --train_data_pattern '/home/vmagent/app/dataset/criteo/valid/test_data.bin' \
  --eval_data_pattern '/home/vmagent/app/dataset/criteo/valid/test_data.bin' \
  --model_dir /home/vmagent/app/e2eaiok/modelzoo/upm/model/pytorch/model.pth \
  --dataset_meta_file pt/data/criteo_meta.yaml \
  --global_batch_size 1024 \
  --eval_batch_size 1024 \
  --num_epochs 2 \
  --learning_rate 0.00048 \
  --platform pytorch