#!/bin/bash

seed=74443
output_folder="results/transformer/$seed"
# wer_file="$output_folder/wer.txt"
save_folder="$output_folder/save"

# pretrained_lm_tokenizer_path="speechbrain/asr-transformer-transformerlm-librispeech"

# Data files
data_folder="/home/vmagent/app/dataset/LibriSpeech"
# train_splits="train-clean-100 train-clean-360 train-other-500"
# dev_splits="dev-clean"
# test_splits="test-clean test-other"
skip_prep=false
train_csv="$data_folder/train-clean-100.csv"
valid_csv="$data_folder/dev-clean.csv"
test_csv="$data_folder/test-clean.csv $data_folder/test-other.csv"

lm_model_ckpt="$save_folder/lm.ckpt"
tokenizer_ckpt="$save_folder/tokenizer.ckpt"

args="--seed $seed \
  --output_folder $output_folder \
  --save_folder $save_folder \
  --data_folder $data_folder \
  --skip_prep $skip_prep \
  --train_csv $train_csv \
  --valid_csv $valid_csv \
  --test_csv $test_csv \
  --lm_model_ckpt $lm_model_ckpt \
  --tokenizer_ckpt $tokenizer_ckpt"

python train.py --param_file config/transformer.yaml --device=cpu $args
# python -m intel_extension_for_pytorch.cpu.launch --use_logical_core train.py --param_file config/transformer.yaml --device=cpu $args
# python -m torch.distributed.launch --nproc_per_node 2 train.py --param_file config/transformer.yaml --device=cpu $args --distributed_launch=True --distributed_backend=gloo
