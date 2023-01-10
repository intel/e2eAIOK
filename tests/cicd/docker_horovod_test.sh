#!/bin/bash

# init conda
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# enable oneAPI
source /opt/intel/oneapi/setvars.sh --force
# UT check for docker env with horovod
cd /home/vmagent/app/e2eaiok
conda activate tensorflow
horovodrun -n 2 -H localhost:2 pytest -v --capture=tee-sys tests/cicd/src/test_docker_horovod.py