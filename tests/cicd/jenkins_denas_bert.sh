#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate pytorch-1.10.0

# launch denas for bert-based supernet
cd /home/vmagent/app/hydro.ai/DeNas
sed -i '/max_epochs:/ s/:.*/: 1/' ../conf/denas/nlp/aidk_denas_bert.conf
python -u search.py --domain bert --conf ../conf/denas/nlp/aidk_denas_bert.conf
cd /home/vmagent/app/hydro.ai

# test
LANG=C tests/cicd/bats/bin/bats tests/cicd/test_denas.bats
