#!/bin/bash
set -x

cd /home/vmagent/app/e2eaiok/e2eAIOK/deltatuner
pip install -e .

cd /home/vmagent/app/e2eaiok

mkdir -p log models

DATA_PATH="/home/vmagent/app/data"

# fine-tune mpt-7b with denas-lora
python example/instruction_tuning_pipeline/finetune_clm.py \
    --model_name_or_path $DATA_PATH"/mpt-7b" \
    --train_file $DATA_PATH"/alpaca_data.json" \
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
    --output_dir models/mpt_denas-lora_model \
    --peft lora \
    --delta lora \
    --debugs --max_epochs 1 --population_num 1 --crossover_num 1 --mutation_num 1 --select_num 1 \
    --trust_remote_code True \
    --no_cuda \
    --bf16 True 2>&1 | tee log/mpt-denas-lora-run-1epoch.log

# fine-tune llama2-7b with denas-lora
python example/instruction_tuning_pipeline/finetune_clm.py \
    --model_name_or_path $DATA_PATH"/Llama-2-7b-hf" \
    --train_file $DATA_PATH"/alpaca_data.json" \
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
    --output_dir models/llama2_denas-lora_model \
    --peft lora \
    --delta lora \
    --debugs --max_epochs 1 --population_num 1 --crossover_num 1 --mutation_num 1 --select_num 1 \
    --trust_remote_code True \
    --no_cuda \
    --bf16 True 2>&1 | tee log/llama2-denas-lora-run-1epoch.log
