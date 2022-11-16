#!/bin/bash

python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 --hostfile hosts \
    trainer/train.py --domain=asr --conf=./asr/config/trainer_config.yaml --param_file config/transformer.yaml --device=cpu --distributed_launch=true --distributed_backend=ccl 2>&1 | tee asr_distributed_training.log

