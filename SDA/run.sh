# WnD

/root/sw/miniconda3/envs/wd2/bin/python main.py --train_path "/mnt/sdd/outbrain2/tfrecords/train/part*" --eval_path "/mnt/sdd/outbrain2/tfrecords/eval/part*" \
--dataset_meta_path /outbrain2/tfrecords --model WnD --dataset_format TFRecords \
--python_executable /root/sw/miniconda3/envs/wd2/bin/python --ppn 2 --ccl_worker_num 2 --metric MAP --metric_threshold 0.6553 --num_epochs 8 --global_batch_size 524288 \
--training_time_threshold 1800 \
--deep_dropout 0.1 \
--observation_budget 3