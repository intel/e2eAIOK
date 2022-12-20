#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate pytorch

# reinstall e2eaiok from source
cd /home/vmagent/app/e2eaiok
python setup.py sdist && pip install dist/e2eAIOK-*.*.*.tar.gz

# launch denas for vit-based supernet
cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas
sed -i '/max_epochs:/ s/:.*/: 1/' ../../conf/denas/cv/e2eaiok_denas_vit.conf
sed -i '/population_num:/ s/:.*/: 1/' ../../conf/denas/cv/e2eaiok_denas_vit.conf
sed -i '/crossover_num:/ s/:.*/: 1/' ../../conf/denas/cv/e2eaiok_denas_vit.conf
sed -i '/mutation_num:/ s/:.*/: 1/' ../../conf/denas/cv/e2eaiok_denas_vit.conf
python -u search.py --domain vit --conf ../../conf/denas/cv/e2eaiok_denas_vit.conf
cd /home/vmagent/app/e2eaiok

# test
LANG=C tests/cicd/bats/bin/bats tests/cicd/test_denas.bats

# train
cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas
sed -i '/train_epochs:/ s/:.*/: 1/' ../../conf/denas/cv/e2eaiok_denas_train_vit.conf
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 train.py --domain vit --conf /home/vmagent/app/e2eaiok/conf/denas/cv/e2eaiok_denas_train_vit.conf