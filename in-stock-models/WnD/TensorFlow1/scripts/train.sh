export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
# export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
# export KMP_BLOCKTIME=30
export OMP_NUM_THREADS=36
# export KMP_SETTINGS=1
set -x
set -e

time /root/sw/miniconda3/envs/wd/bin/python -m trainer.task \
  --train_data_pattern hdfs://sr112:9001/outbrain/tfrecords/train/part* \
  --eval_data_pattern hdfs://sr112:9001/outbrain/tfrecords/eval/part* \
  --model_dir /mnt/nvm6/wd/checkpoints \
  --transformed_metadata_path /outbrain/tfrecords \
  --deep_hidden_units 1024 512 256