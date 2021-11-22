export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
# export KMP_AFFINITY=granularity=fine,compact,1,0
# export KMP_BLOCKTIME=0
# export OMP_NUM_THREADS=18
# export KMP_SETTINGS=TRUE
set -x
set -e

# Intel
# mpirun -x OMP_NUM_THREADS=18 --allow-run-as-root -cpus-per-proc 18 --map-by node --report-bindings  -np 4 -H sr112:2,sr113:2 \
#   /root/sw/miniconda3/envs/wd/bin/python -m trainer.task \
#   --train_data_pattern hdfs://sr112:9001/outbrain/tfrecords/train/part* \
#   --eval_data_pattern hdfs://sr112:9001/outbrain/tfrecords/eval/part* \
#   --model_dir /mnt/nvm6/wd/checkpoints \
#   --transformed_metadata_path /outbrain/tfrecords \
#   --hvd

  # time mpirun -x OMP_NUM_THREADS=18 --allow-run-as-root -cpus-per-proc 18 --map-by socket --report-bindings  -n 2 -H sr112:2 \
  # /root/sw/miniconda3/envs/wd/bin/python -m trainer.task \
  # --train_data_pattern hdfs://sr112:9001/outbrain/tfrecords/train/part* \
  # --eval_data_pattern hdfs://sr112:9001/outbrain/tfrecords/eval/part* \
  # --model_dir /mnt/nvm6/wd/checkpoints \
  # --transformed_metadata_path /outbrain/tfrecords \
  # --hvd

# time horovodrun --mpi --mpi-args="-x OMP_NUM_THREADS=18 -x KMP_WARNINGS=0 --allow-run-as-root -cpus-per-proc 18 --map-by socket --report-bindings --oversubscribe" \
# -np 8 -H sr112:2,sr612:2,sr610:2,sr613:2 --start-timeout 600 --timeline-filename timeline.log \
# /root/sw/miniconda3/envs/wd/bin/python -m trainer.task \
# --train_data_pattern hdfs://sr112:9001/outbrain/tfrecords/train/part* \
# --eval_data_pattern hdfs://sr112:9001/outbrain/tfrecords/eval/part* \
# --model_dir /mnt/nvm6/wd/checkpoints-tmp \
# --transformed_metadata_path /outbrain/tfrecords \
# --hvd \
# --deep_hidden_units 1024 512 256 

time mpirun -x OMP_NUM_THREADS=18 -x KMP_WARNINGS=0 --allow-run-as-root -cpus-per-proc 18 --map-by socket --report-bindings --oversubscribe -np 8 -H sr112:2,sr612:2,sr610:2,sr613:2 \
  /root/sw/miniconda3/envs/wd/bin/python -m trainer.task \
  --train_data_pattern hdfs://sr112:9001/outbrain/tfrecords/train/part* \
  --eval_data_pattern hdfs://sr112:9001/outbrain/tfrecords/eval/part* \
  --model_dir /mnt/nvm6/wd/checkpoints-warmup2 \
  --transformed_metadata_path /outbrain/tfrecords \
  --hvd \
  --deep_hidden_units 1024 512 256

# benchmark performance
# time mpirun -x OMP_NUM_THREADS=18 -x KMP_WARNINGS=0 --allow-run-as-root -cpus-per-proc 18 --map-by socket --report-bindings --oversubscribe -np 8 -H sr112:2,sr612:2,sr610:2,sr613:2 \
#   /root/sw/miniconda3/envs/wd/bin/python -m trainer.task \
#   --train_data_pattern hdfs://sr112:9001/outbrain/tfrecords/train/part* \
#   --eval_data_pattern hdfs://sr112:9001/outbrain/tfrecords/eval/part* \
#   --model_dir /mnt/nvm6/wd/checkpoints-test \
#   --transformed_metadata_path /outbrain/tfrecords \
#   --hvd \
#   --deep_hidden_units 1024 512 256 \
#   --benchmark_warmup_steps 50 \
#   --benchmark_steps 100 \
#   --benchmark
  
