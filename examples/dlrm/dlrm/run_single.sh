DATA_PATH="/mnt/DP_disk7/binary_dataset/"

seed_num=$(date +%s)


python -u /home/xianyang/BlueWhale-poc/mlperf2/dlrm2/dlrm/dlrm_s_pytorch_joint_embedding.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000  --data-generation=dataset --data-set=terabyte --raw-data-file=$DATA_PATH/day --processed-data-file=$DATA_PATH/terabyte_processed.npz --loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex --learning-rate=18.0 --mini-batch-size=32768 --print-freq=128 --print-time --test-freq=6400 --test-mini-batch-size=65536 --lr-num-warmup-steps=8000 --lr-decay-start-step=70000 --lr-num-decay-steps=30000 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=$seed_num $dlrm_extra_option

