## Run specific params
export DATADIR="/raid/datasets/rnnt/"
export METADATA_DIR="/lustre/fsw/mlperf-ci/tokenized/"
export SENTENCEPIECES_DIR="/lustre/fsw/mlperf-ci/sentpiece"
export BATCHSIZE=32
export EVAL_BATCHSIZE=338
export GRAD_ACCUMULATION_STEPS=1
export WALLTIME=04:00:00
export MAX_SYMBOL=300
export DATA_CPU_THREADS=4

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_2048.sh

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
# export BATCH_EVAL_MODE=cg_unroll_pipeline
export APEX_LOSS=fp16
export APEX_JOINT=pack

export AMP_LVL=1
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=false
export MULTILAYER_LSTM=true
export ENABLE_PREFETCH=true
export BATCH_SPLIT_FACTOR=1
export TOKENIZED_TRANSCRIPT=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export MIN_SEQ_SPLIT_LEN=20
export APEX_MLP=true
export PRE_SORT_FOR_SEQ_SPLIT=true
export LOG_FREQUENCY=1
export JIT_TENSOR_FORMATION=true