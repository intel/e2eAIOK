#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch
cd /home/vmagent/app/e2eaiok
python setup.py sdist && pip install dist/e2eAIOK-*.*.*.tar.gz
# run main.py
python /home/vmagent/app/e2eaiok/e2eAIOK/ModelAdapter/main.py --cfg /home/vmagent/app/e2eaiok/conf/ma/demo/distiller/cifar100_kd_res50_res18.yaml