set -x

cd /home/vmagent/app/e2eaiok/e2eAIOK/deltatuner
pip install -e .
pip uninstall wandb -y

cd /home/vmagent/app/e2eaiok

DATA_PATH="/home/vmagent/app/data"
LOG_PATH=$DATA_PATH"/dtuner_test/log"
MODEL_SAVE_PATH=$DATA_PATH"/dtuner_test/models"

mkdir -p $LOG_PATH $MODEL_SAVE_PATH

# fine-tune with denas-ssf
model_name_list="mpt-7b Llama-2-7b-hf"
for model_name in $model_name_list
do
    model_name_or_path=${DATA_PATH}"/"${model_name}
    model_save_path=${MODEL_SAVE_PATH}"/"${model_name}"_denas-ssf"
    log_save_path=$LOG_PATH"/"${model_name}"_denas-ssf-1epoch.log"
    python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path $model_name_or_path \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_train \
        --do_eval \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --trust_remote_code True \
        --no_cuda \
        --output_dir $model_save_path \
        --debugs --max_epochs 1 --population_num 1 --crossover_num 1 --mutation_num 1 --select_num 1 \
        --algo ssf \
        --denas True \
        2>&1 | tee $log_save_path
done
