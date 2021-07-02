source ~/.local/env/setvars.sh
seed_num=$(date +%s)
#export CCL_ATL_SHM=0
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"


#Four nodes
python  -u  ./launch.py --distributed --nproc_per_node=2 --nnodes=4 --hostfile ./hosts --master_addr="10.x.x.xxx" /home/xianyang/BlueWhale-poc/mlperf2/dlrm2/dlrm/dlrm_s_pytorch_lamb_sparselamb_test.py --arch-sparse-feature-size=64 --arch-mlp-bot="13-128-64" --arch-mlp-top="256-128-1" --max-ind-range=40000000  --data-generation=dataset --data-set=terabyte --raw-data-file=$DATA_PATH/day --processed-data-file=$DATA_PATH/terabyte_processed.npz --loss-function=bce --round-targets=True --bf16 --num-workers=0 --test-num-workers=0 --use-ipex  --optimizer=1 --dist-backend=ccl --learning-rate=16 --mini-batch-size=262144 --print-freq=16 --print-time --test-freq=800 --sparse-dense-boundary=403346 --test-mini-batch-size=16384 --lr-num-warmup-steps=4000 --enable-profiling --lr-decay-start-step=5760 --lr-num-decay-steps=27000 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=12345 $dlrm_extra_option 2>&1 | tee run_${seed_num}_03_30_2021.log
