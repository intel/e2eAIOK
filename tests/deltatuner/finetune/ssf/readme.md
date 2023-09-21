# Run Deltatuner with ssf

## Llama2-7b

### 1. Finetune the model
If use Denas, can use command:

```shell
cd frameworks.bigdata.AIDK/example
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
```

### 2. Save merged model
Merge the SSF adapter into orginal base model.

```shell
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
```

### 3. Evaluate merged model
As ssf will enable bias, while the default Llama2 disable all the bias, to enable the full parameters of the adapter, we need to change the model definition.

First copy the updated model codes with the merged weights.
```shell
cp instruction_tuning_pipeline/models/llama2-ssf/* $DATA_PATH/llama2-7b-ssf-denas-bf16-merge/merged_model
```

Then update the "best_model_structure" and "target_modules" setting in config.json. if not enable denas and not change "target_modules" default settings, can skip correpsonding setting.
```shell
...
"best_model_structure":  {"num_hidden_layers": [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]}, # change to your best structure, skip to keep default
"target_modules": ["q_proj", "v_proj"], #change to your setting, skip to keep default
...
```

Finally we can directly evalute the merged model.
```shell
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
        --bf16 True
```