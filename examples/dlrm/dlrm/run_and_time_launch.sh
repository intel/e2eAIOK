source ~/.local/env/setvars.sh

seed_num=$(date +%s)
#export CCL_ATL_SHM=0
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

python  -u  ./launch.py --distributed --nproc_per_node=2 --nnodes=4 --hostfile ./hosts --master_addr="xxxxx" dlrm_s_pytorch.py  --max-ind-range=40000000  --data-generation=dataset --data-set=terabyte --raw-data-file=$DATA_PATH/day --processed-data-file=$DATA_PATH/terabyte_processed.npz --loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex  --test-mini-batch-size=65536 --dist-backend=ccl  --mini-batch-size=262144 --print-freq=16 --print-time --optimizer=1 --bf16 --test-freq=800 --sparse-dense-boundary=403346  --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=12345 $dlrm_extra_option 2>&1 | tee run_terabyte_${seed_num}.log
