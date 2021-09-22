set -x
set -e

 export CCL_WORKER_COUNT=2
 export CCL_WORKER_AFFINITY="16,17,34,35"
 export HOROVOD_THREAD_AFFINITY="53,71"
 export I_MPI_PIN_DOMAIN=socket
 export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="16,17,34,35,52,53,70,71"


time mpirun -genv OMP_NUM_THREADS=16 -map-by socket -n 2 -ppn 2 -hosts sr112 -print-rank-map \
-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 \
-iface eth3 \
/root/sw/miniconda3/envs/wd2/bin/python sigopt_runner.py \
  --train_data_pattern /mnt/sdd/outbrain2/tfrecords/train/part* \
  --eval_data_pattern /mnt/sdd/outbrain2/tfrecords/eval/part* \
  --model_dir /mnt/nvm6/wd/checkpoints2 \
  --transformed_metadata_path /outbrain2/tfrecords \
  --global_batch_size 524288 \
  --eval_batch_size 524288 \
  --num_epochs 50 \
  --deep_learning_rate 0.00048 \
  --linear_learning_rate 0.8 \
  --deep_hidden_units 1024 512 256 \
  --deep_dropout 0.1
