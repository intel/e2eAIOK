#!/bin/bash
set -x

mkdir -p log models

# lora on mpt
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/home/vmagent/app/data/mpt-7b" \
        --train_file "/home/vmagent/app/data/stanford_alpaca/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --do_eval \
        --validation_split_percentage 30 \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --output_dir models/mpt_lora_finetuned_model \
        --peft lora \
        --lora_target_modules Wqkv out_proj up_proj down_proj \
        --trust_remote_code True \
        --no_cuda \
        --bf16 True 2>&1 | tee log/mpt-lora-run-1epoch.log

# delta-lora on mpt
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/home/vmagent/app/data/mpt-7b" \
        --train_file "/home/vmagent/app/data/stanford_alpaca/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --do_eval \
        --validation_split_percentage 30 \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --output_dir models/mpt_delta_finetuned_model \
        --peft lora \
        --lora_target_modules Wqkv out_proj up_proj down_proj \
        --delta True \
        --denas True \
        --trust_remote_code True \
        --no_cuda \
        --bf16 True 2>&1 | tee log/mpt-detla-run-1epoch.log
