DATA_PATH="/home/vmagent/app/data"


# merge model
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/Llama-2-7b-hf" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_eval \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --trust_remote_code True \
        --no_cuda \
        --output_dir "$DATA_PATH/dtuner_test/models/llama2-7b-ssf-denas-bf16-merge" \
        --algo ssf \
        --resume_peft "$DATA_PATH/dtuner_test/models/llama2-7b-ssf-denas-bf16" \
        --save_merged_model True \
        --merge_model_code_dir $DATA_PATH"/dtuner_test/ssf_models_code/llama-7b-ssf" \
        --bf16 True \
        --debugs


# evaluate merged model
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path $DATA_PATH"/dtuner_test/models/llama2-7b-ssf-denas-bf16-merge/merged_model" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_eval \
        --save_total_limit 1 \
        --log_level info \
        --trust_remote_code True \
        --no_cuda \
        --bf16 True \
        --output_dir $DATA_PATH"/dtuner_test/models/llama2-7b-ssf-denas-bf16-merge/eval_merge" \
        --debugs

