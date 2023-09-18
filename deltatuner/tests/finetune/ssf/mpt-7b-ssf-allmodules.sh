DATA_PATH="/home/data"

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
        --output_dir "$DATA_PATH/mpt-7b-ssf-allmodules" \
        --delta ssf \
        --denas False \
        --ssf_target_modules Wqkv out_proj up_proj down_proj \
        | tee mpt-ssf-run-allmodules-1epoch.log

# bf16
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
        --output_dir "$DATA_PATH/mpt-7b-ssf-allmodules-bf16" \
        --delta ssf \
        --denas False \
        --ssf_target_modules Wqkv out_proj up_proj down_proj \
        --bf16 True \
        | tee mpt-ssf-run-allmodules-bf16-1epoch.log

# merge, bf16
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/mpt-7b" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
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
        --output_dir "$DATA_PATH/mpt-7b-ssf-allmodules-bf16-merge" \
        --delta ssf \
        --ssf_target_modules Wqkv out_proj up_proj down_proj \
        --bf16 True \
        --resume_peft "$DATA_PATH/mpt-7b-ssf-allmodules-bf16" \
        --save_merged_model True

#evaluate merged model, bf16
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/mpt-7b-ssf-allmodules-bf16-merge/merged_model" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_eval \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --trust_remote_code True \
        --tokenizer_name "$DATA_PATH/gpt-neox-20b" \
        --no_cuda \
        --output_dir "$DATA_PATH/mpt-7b-ssf-allmodules-bf16-merge/eval_merged" \
        --bf16 True \
        --debugs
