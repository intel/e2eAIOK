TOKEN=${1}
# WnD

# outbrain

SIGOPT_API_TOKEN=${TOKEN} python main.py --train_path "/home/vmagent/app/dataset/outbrain/tfrecords/train/part*" --eval_path "/home/vmagent/app/dataset/outbrain/tfrecords/eval/part*" \
--dataset_meta_path /home/vmagent/app/dataset/outbrain/outbrain_meta.yaml --model WnD \
--python_executable /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python --ppn 2 --ccl_worker_num 2 --metric MAP --metric_threshold 0.6553 --num_epochs 8 --global_batch_size 524288 \
--training_time_threshold 1800 \
--program /home/vmagent/app/bluewhale/examples/WnD/TensorFlow2/main.py \
--deep_dropout 0.1 \
--observation_budget 1

# criteo

# SIGOPT_API_TOKEN=${TOKEN} python main.py --train_path "/home/vmagent/app/dataset/criteo/train_data.bin" --eval_path "/home/vmagent/app/dataset/criteo/test_data.bin" \
# --dataset_meta_path /home/vmagent/app/dataset/criteo/criteo_meta.yaml --model WnD \
# --python_executable /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python --ppn 2 --ccl_worker_num 2 --metric AUC --metric_threshold 0.6553 --num_epochs 1 --global_batch_size 524288 \
# --training_time_threshold 1800 \
# --program /home/vmagent/app/bluewhale/examples/WnD/TensorFlow2/main.py \
# --deep_warmup_epochs 1 \
# --observation_budget 3

# DLRM

#export CCL_ATL_SHM=0
#export KMP_BLOCKTIME=1
#export KMP_AFFINITY="granularity=fine,compact,1,0"

# SIGOPT_API_TOKEN=${TOKEN} seed_num=$(date +%s) python main.py --train_path /home/vmagent/app/dataset/criteo/train_data.bin --eval_path /home/vmagent/app/dataset/criteo/test_data.bin \
# --dataset_meta_path /home/vmagent/app/dataset/criteo/criteo_meta.yaml --model DLRM \
# --python_executable /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python --ppn 2 --metric AUC --metric_threshold 0.8025 \
# --hosts localhost
