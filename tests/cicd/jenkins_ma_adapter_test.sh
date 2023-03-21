#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch
cd /home/vmagent/app/e2eaiok
python setup.py sdist && pip install dist/e2eAIOK-*.*.*.tar.gz
# run main.py
cd /home/vmagent/app/e2eaiok/modelzoo/unet
sh patch_unet.sh
sh scripts/run_single_demo.sh