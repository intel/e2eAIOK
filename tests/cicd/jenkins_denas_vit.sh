#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate pytorch

# launch denas for vit-based supernet
cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas
sed -i '/max_epochs:/ s/:.*/: 1/' ../../conf/denas/cv/e2eaiok_denas_vit.conf
python -u search.py --domain vit --conf ../../conf/denas/cv/e2eaiok_denas_vit.conf
cd /home/vmagent/app/e2eaiok

# test
LANG=C tests/cicd/bats/bin/bats tests/cicd/test_denas.bats

cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 train.py --domain vit --conf /home/vmagent/app/e2eaiok/e2eAIOK/conf/denas/cv/e2eaiok_denas_train_vit.conf