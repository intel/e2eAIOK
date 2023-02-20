#!/bin/bash

docker exec -it aiok-test bash -c "source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force && \
conda activate pytorch-1.12.0 && \
cd /home/vmagent/app/e2eAIOK/modelzoo/unet && \
time sh scripts/run_dist_opt.sh vsr257,vsr262,vsr263,vsr264 4 3 && \
time sh scripts/run_predict.sh"