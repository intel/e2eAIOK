#!/bin/bash
set -x

cd /home/vmagent/app/e2eaiok/e2eAIOK/deltatuner
pip install -e .
pip uninstall wandb -y

cd /home/vmagent/app/e2eaiok

DATA_PATH="/home/vmagent/app/data"
LOG_PATH=$DATA_PATH"/dtuner_test/log"
MODEL_SAVE_PATH=$DATA_PATH"/dtuner_test/models"

mkdir -p $LOG_PATH $MODEL_SAVE_PATH

# fine-tune with denas-lora
model_name_list="mpt-7b Llama-2-7b-hf"
for model_name in $model_name_list
do
    model_name_or_path=${DATA_PATH}"/"${model_name}
    model_save_path=${MODEL_SAVE_PATH}"/"${model_name}"_denas-lora"
    log_save_path=$LOG_PATH"/"${model_name}"_denas-lora-1epoch.log"
    python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path $model_name_or_path \
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
        --output_dir $model_save_path \
        --peft lora \
        --algo lora \
        --denas True \
        --debugs --max_epochs 1 --population_num 1 --crossover_num 1 --mutation_num 1 --select_num 1 \
        --trust_remote_code True \
        --no_cuda \
        2>&1 | tee $log_save_path
done
