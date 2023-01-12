#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.12.0
# run unittest
cd /home/vmagent/app/e2eaiok/e2eAIOK/ModelAdapter/test

# pytest for single test file
# pytest -v adapter_test.py 
# pytest -v basic_finetunner_test.py

# pytest all unittest
python run_testting.py