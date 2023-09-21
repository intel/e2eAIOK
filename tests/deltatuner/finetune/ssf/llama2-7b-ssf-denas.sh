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
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas" \
        --delta ssf \
        --denas True \
        | tee llama2-7b-ssf-denas-1epoch.log

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
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas-bf16" \
        --delta ssf \
        --denas True \
        --bf16 True \
        | tee llama2-7b-ssf-denas-bf16-1epoch.log

#evaluate model, bf16
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/Llama-2-7b-hf" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_eval \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --trust_remote_code True \
        --no_cuda \
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas-bf16/eval" \
        --delta ssf \
        --bf16 True \
        --resume_peft "$DATA_PATH/llama2-7b-ssf-denas-bf16" \
        --denas "$DATA_PATH/llama2-7b-ssf-denas-bf16/best_model_structure.txt" \
        --debugs

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
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas-merge" \
        --delta ssf \
        --resume_peft "$DATA_PATH/llama2-7b-ssf-denas" \
        --save_merged_model True \
        --denas "$DATA_PATH/llama2-7b-ssf-denas/best_model_structure.txt" \
        --debugs

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
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas-bf16-merge" \
        --delta ssf \
        --bf16 True \
        --resume_peft "$DATA_PATH/llama2-7b-ssf-denas-bf16" \
        --save_merged_model True \
        --denas "$DATA_PATH/llama2-7b-ssf-denas-bf16/best_model_structure.txt" \
        --debugs

#evaluate merged model
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/llama2-7b-ssf-denas-merge/merged_model" \
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
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas-merge/eval_merge" \
        --debugs

#evaluate merged model, bf16
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/llama2-7b-ssf-denas-bf16-merge/merged_model" \
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
        --output_dir "$DATA_PATH/llama2-7b-ssf-denas-bf16-merge/eval_merge" \
        --bf16 True \
        --debugs