#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
eval "$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
cd modelzoo/minigo
conda activate minigo
yes "" | ./cc/configure_tensorflow.sh
./ml_perf/scripts/cc_libgen_parallel_selfplay.sh
cd ../..