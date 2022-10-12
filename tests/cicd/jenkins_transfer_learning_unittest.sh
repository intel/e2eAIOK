#!/bin/bash

# init conda env
eval "$('/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# activate conda
conda activate pytorch-1.10.0
# run unittest
cd /home/vmagent/app/AIDK/TransferLearningKit/test
pytest -v adapter_test.py # pytest for single test file
# pytest -v test_basic_finetunner.py
python run_testting.py