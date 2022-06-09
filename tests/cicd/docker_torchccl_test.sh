#!/bin/bash

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# UT check for docker env with torchccl
cd /home/vmagent/app/hydro.ai
conda activate pytorch_1.10
mpirun -n 2 -l pytest -v --capture=tee-sys tests/cicd/src/test_docker_torchccl.py