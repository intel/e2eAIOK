#!/bin/bash

# init conda
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# UT check for docker env with torchccl
source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh
cd /home/vmagent/app/e2eaiok
conda activate pytorch_mlperf
mpirun -n 2 -l pytest -v --capture=tee-sys tests/cicd/src/test_docker_torchccl.py