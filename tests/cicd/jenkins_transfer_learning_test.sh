#!/bin/bash

# set vars
# MODEL_NAME="TLK"
# DATA_PATH="/home/vmagent/app/AIDK/TransferLearningKit/datasets"

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.10.0
echo `pwd`
ls
# run main.py
cd /home/vmagent/app/AIDK/TransferLearningKit/src
python main.py