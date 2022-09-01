#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate pytorch-1.10.0

# launch denas for vit-based supernet
cd /home/vmagent/app/hydro.ai/DeNas
sed -i '/max_epochs:/ s/:.*/: 1/' ../conf/denas/cv/aidk_denas_vit.conf
python -u search.py --domain vit --conf ../conf/denas/cv/aidk_denas_vit.conf
cd /home/vmagent/app/hydro.ai

# test
LANG=C tests/cicd/bats/bin/bats tests/cicd/test_denas.bats