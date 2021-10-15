# WnD

# outbrain

/root/sw/miniconda3/envs/wd2/bin/python main.py --train_path "/mnt/sdd/outbrain2/tfrecords/train/part*" --eval_path "/mnt/sdd/outbrain2/tfrecords/eval/part*" \
--dataset_meta_path /root/ht/ML/frameworks.bigdata.bluewhale/examples/WnD/TensorFlow2/data/outbrain/outbrain_meta.yaml --model WnD \
--python_executable /root/sw/miniconda3/envs/wd2/bin/python --ppn 2 --ccl_worker_num 2 --metric MAP --metric_threshold 0.6553 --num_epochs 8 --global_batch_size 524288 \
--training_time_threshold 1800 \
--program /root/ht/ML/frameworks.bigdata.bluewhale/examples/WnD/TensorFlow2/main.py \
--deep_dropout 0.1 \
--observation_budget 3

# criteo

# /root/sw/miniconda3/envs/wd2/bin/python main.py --train_path "/mnt/nvm6/criteo/train_data.bin" --eval_path "/mnt/nvm6/criteo/test_data.bin" \
# --dataset_meta_path /root/ht/ML/frameworks.bigdata.bluewhale/examples/WnD/TensorFlow2/data/outbrain/criteo_meta.yaml --model WnD \
# --python_executable /root/sw/miniconda3/envs/wd2/bin/python --ppn 2 --ccl_worker_num 2 --metric AUC --metric_threshold 0.6553 --num_epochs 1 --global_batch_size 524288 \
# --training_time_threshold 1800 \
# --program /root/ht/ML/frameworks.bigdata.bluewhale/examples/WnD/TensorFlow2/main.py \
# --deep_warmup_epochs 1 \
# --observation_budget 3