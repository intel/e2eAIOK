#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.10.0
# run main.py
cd /home/vmagent/app/AIDK/TransferLearningKit/src
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 main.py --cfg ../config/demo/baseline/cifar100_res18.yaml