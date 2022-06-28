#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate pytorch_1.10

# launch denas for cnn-based supernet
cd /home/vmagent/app/hydro.ai/DeNas
python -u search.py --domain cnn --conf ../conf/denas/cv/aidk_denas_cnn.conf
cd /home/vmagent/app/hydro.ai

# test
LANG=C tests/cicd/bats/bin/bats tests/cicd/test_denas_cnn.bats