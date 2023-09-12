DATA_PATH="/home/data"

#bf16
python instruction_tuning_pipeline/finetune_clm.py \
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
        --tokenizer_name "$DATA_PATH/gpt-neox-20b" \
        --no_cuda \
        --output_dir "$DATA_PATH/mpt-7b-ssf-allmodules-denas-bf16" \
        --delta ssf \
        --ssf_target_modules Wqkv out_proj up_proj down_proj \
        --bf16 True \
        --denas True \
        | tee "mpt-7b-ssf-allmodules-denas-bf16-run-1epoch.log"