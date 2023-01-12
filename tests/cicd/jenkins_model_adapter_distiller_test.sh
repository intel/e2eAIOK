#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.12.0
# run main.py
cd /home/vmagent/app/e2eaiok/e2eAIOK/ModelAdapter/src
python main.py --cfg ../config/demo/distiller/cifar100_kd_res50_res18.yaml