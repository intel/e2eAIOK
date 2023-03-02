#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate pytorch

# reinstall e2eaiok from source
cd /home/vmagent/app/e2eaiok
python setup.py sdist && pip install dist/e2eAIOK-*.*.*.tar.gz

# launch denas for hf supernet
cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas
sed -i '/max_epochs:/ s/:.*/: 1/' ../../conf/denas/hf/e2eaiok_denas_hf.conf
sed -i '/population_num:/ s/:.*/: 1/' ../../conf/denas/hf/e2eaiok_denas_hf.conf
sed -i '/crossover_num:/ s/:.*/: 1/' ../../conf/denas/hf/e2eaiok_denas_hf.conf
sed -i '/mutation_num:/ s/:.*/: 1/' ../../conf/denas/hf/e2eaiok_denas_hf.conf
python -u search.py --domain hf --conf ../../conf/denas/hf/e2eaiok_denas_hf.conf
cd /home/vmagent/app/e2eaiok

# test
LANG=C tests/cicd/bats/bin/bats tests/cicd/test_denas.bats