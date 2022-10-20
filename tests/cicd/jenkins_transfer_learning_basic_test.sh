#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.10.0
# run main.py
cd /home/vmagent/app/AIDK/TransferLearningKit/src
python main.py -s1 -r0 --cfg ../config/baseline/cifar100_resnet18_test.yaml