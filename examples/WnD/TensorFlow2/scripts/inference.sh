set -x
set -e
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
# CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
# export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
# export KMP_BLOCKTIME=30
# export OMP_NUM_THREADS=36
# export KMP_SETTINGS=1

 export CCL_WORKER_COUNT=2
 export CCL_WORKER_AFFINITY="16,17,34,35"
 export HOROVOD_THREAD_AFFINITY="53,71"
 export I_MPI_PIN_DOMAIN=socket
 export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="16,17,34,35,52,53,70,71"
# export HOROVOD_CCL_CACHE=1

# time mpirun -genv OMP_NUM_THREADS=16 -map-by socket -n 2 -ppn 2 -hosts sr113 -print-rank-map \
# -genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 \
# /root/sw/miniconda3/envs/wd2/bin/python main.py \
#   --train_data_pattern /mnt/sdd/outbrain2/tfrecords/train/part* \
#   --eval_data_pattern /mnt/sdd/outbrain2/tfrecords/eval/part* \
#   --model_dir /mnt/nvm6/wd/checkpoints2 \
#   --transformed_metadata_path /outbrain2/tfrecords \
#   --cpu \
#   --global_batch_size 131072

time mpirun -genv OMP_NUM_THREADS=16 -map-by socket -n 8 -ppn 2 -hosts sr113,sr610,sr612,sr613 -print-rank-map \
-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 \
/root/sw/miniconda3/envs/wd2/bin/python main.py \
  --train_data_pattern /mnt/sdd/outbrain2/tfrecords/train/part* \
  --eval_data_pattern /mnt/sdd/outbrain2/tfrecords/eval/part* \
  --model_dir /mnt/nvm6/wd/checkpoints2-infer-test \
  --transformed_metadata_path /outbrain2/tfrecords \
  --cpu \
  --deep_learning_rate 0.00048 \
  --linear_learning_rate 0.8 \
  --eval_batch_size 1048576 \
  --evaluate \
  --use_checkpoint \
  --benchmark \
  --benchmark_warmup_steps 50 \
  --benchmark_steps 100