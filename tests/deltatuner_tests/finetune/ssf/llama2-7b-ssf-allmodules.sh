DATA_PATH="/home/data"

python instruction_tuning_pipeline/finetune_clm.py \
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
        --output_dir "$DATA_PATH/llama2-7b-ssf-allmodules" \
        --ssf_target_modules q_proj v_proj k_proj o_proj up_proj down_proj \
        --delta ssf \
        --denas False \
        | tee llama2-7b-ssf-allmodules-1epoch.log

#bf16  
python instruction_tuning_pipeline/finetune_clm.py \
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
        --output_dir "$DATA_PATH/llama2-7b-ssf-allmodules-bf16" \
        --ssf_target_modules q_proj v_proj k_proj o_proj up_proj down_proj \
        --delta ssf \
        --bf16 True \
        --denas False \
        | tee llama2-7b-ssf-allmodules-bf16-1epoch.log

#merge
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/Llama-2-7b-hf" \
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
        --no_cuda \
        --output_dir "$DATA_PATH/llama2-7b-ssf-allmodules-merge" \
        --ssf_target_modules q_proj v_proj k_proj o_proj up_proj down_proj \
        --delta ssf \
        --resume_peft "$DATA_PATH/llama2-7b-ssf-allmodules" \
        --save_merged_model True

#merge, bf16
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/Llama-2-7b-hf" \
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
        --no_cuda \
        --output_dir "$DATA_PATH/llama2-7b-ssf-allmodules-bf16-merge" \
        --ssf_target_modules q_proj v_proj k_proj o_proj up_proj down_proj \
        --delta ssf \
        --resume_peft "$DATA_PATH/llama2-7b-ssf-allmodules-bf16" \
        --save_merged_model True \
        --bf16 True 

#evaluate merged model
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/llama2-7b-ssf-allmodules-merge/merged_model" \
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
        --no_cuda \
        --output_dir "$DATA_PATH/llama2-7b-ssf-allmodules-merge/eval_merged" \
        --debugs

#evaluate merged model, bf16
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/llama2-7b-ssf-allmodules-bf16-merge/merged_model" \
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
        --no_cuda \
        --output_dir "$DATA_PATH/llama2-7b-ssf-allmodules-bf16-merge/eval_merged" \        
        --bf16 True \
        --debugs