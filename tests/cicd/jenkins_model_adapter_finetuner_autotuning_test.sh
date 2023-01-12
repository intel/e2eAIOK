#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.12.0
# run main.py
cd /home/vmagent/app/e2eaiok/e2eAIOK/ModelAdapter/src
python main.py -R1 --cfg ../config/demo/finetuner/cifar100_res50PretrainI21k.yaml
