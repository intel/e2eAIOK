#!/bin/bash

# enable oneAPI
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# UT check for docker env with horovod
cd /home/vmagent/app/e2eaiok
conda activate tensorflow
horovodrun -n 2 -H localhost:2 pytest -v --capture=tee-sys tests/cicd/src/test_docker_horovod.py