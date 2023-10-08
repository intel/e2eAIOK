set -x

cd /home/vmagent/app/e2eaiok/e2eAIOK/deltatuner
pip install -e .
pip uninstall wandb -y

cd /home/vmagent/app/e2eaiok

DATA_PATH="/home/vmagent/app/data"
LOG_PATH=$DATA_PATH"/dtuner_test/log"
MODEL_SAVE_PATH=$DATA_PATH"/dtuner_test/models"

mkdir -p $LOG_PATH $MODEL_SAVE_PATH

#run mpt with ssf, bf16
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/mpt-7b" \
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
        --output_dir $MODEL_SAVE_PATH"/mpt-7b-ssf-allmodules" \
        --delta ssf \
        --denas False \
        --debugs \
        2>&1 | tee $LOG_PATH"/mpt-7b-ssf-allmodules-1epoch.log"
    
#run llama2 with ssf, bf16
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/Llama-2-7b-hf" \
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
        --output_dir $MODEL_SAVE_PATH"/llama2-7b-ssf" \
        --delta ssf \
        --denas False \
        --debugs \
        2>&1 | tee $LOG_PATH"/llama2-7b-ssf-1epoch.log"