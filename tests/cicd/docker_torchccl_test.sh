#!/bin/bash

# init conda
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# UT check for docker env with torchccl
cd /home/vmagent/app/hydro.ai
conda activate pytorch-1.10.0
mpirun -n 2 -l pytest -v --capture=tee-sys tests/cicd/src/test_docker_torchccl.py