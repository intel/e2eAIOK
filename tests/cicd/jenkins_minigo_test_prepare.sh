#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
eval "$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
cd modelzoo/minigo
conda activate minigo
printf "\n" | ./cc/configure_tensorflow.sh
export HOME=/root
./ml_perf/scripts/cc_libgen_parallel_selfplay.sh
cd ../..