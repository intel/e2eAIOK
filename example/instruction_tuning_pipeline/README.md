# Deltatuner Fine-tuning Example

This example demonstrates how to finetune the pretrained large language model (LLM) with the instruction-following dataset by using the Deltatuner and some foundamental models. Giving Fine-tuned model the textual instruction, it will respond with the textual response. This example have been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids.
This example was based upon [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers/tree/main/workflows/chatbot/fine_tuning).

## Prerequisite​

### 1. Environment​
- build the docker image
```shell
docker build -f Dockerfile-ubuntu/Dockerfile-v1.2 -t chatbot_finetune .
```
- create docker container
```shell
# in cpu server
docker run -it --name chatbot \
    --privileged --network host --ipc=host \
    --device=/dev/dri \
    -v /dev/shm:/dev/shm \
    -v /path/to/code/and/data:/home/vmagent/app/data \
    -w /home/vmagent/app  \
    chatbot_finetune:latest \
    /bin/bash

# in gpu server
docker run -it --name chatbot \
    --privileged --network host --ipc=host \
    --device=/dev/dri \
    --runtime=nvidia \
    -v /dev/shm:/dev/shm \
    -v /path/to/code/and/data:/home/vmagent/app/data \
    -w /home/vmagent/app/  \
    chatbot_finetune:latest \
    /bin/bash
```
- Or you can direct create it from the bare mental env
```shell
pip install deltatuner
```

### 2. Prepare the Model

- MPT: Download [the released model on Huggingface](https://huggingface.co/mosaicml/mpt-7b)
- Llama-7B: Download [the released model on Huggingface](https://huggingface.co/huggyllama/llama-7b)
- Llama-2-7B: Download [the released model on Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)

### 3. Prepare Dataset
The [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). You can also download from [the released dataset on Huggingface](https://huggingface.co/datasets/tatsu-lab/alpaca), or you can download [the cleaned version of Alpaca on Huggingface](https://huggingface.co/datasets/yahma/alpaca-cleaned).


## Finetune

### Lora Fine-tuning with deltatuner

For [MPT](https://huggingface.co/mosaicml/mpt-7b), use the below command line for finetuning on the Alpaca dataset. This model also requires that trust_remote_code=True be passed to the from_pretrained method. This is because we use a custom MPT model architecture that is not yet part of the Hugging Face transformers package.

```bash
# fine-tune with denas-lora
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
    --trust_remote_code True \
    --no_cuda \
    2>&1 | tee $log_save_path
```

- Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

- For model profile, add `--profile` argument, the forward time and backward time will print on the console.

- For target modules of LoRA, use `--lora_target_modules` argument, the default target module of MPT and LLaMa model is `"llama": ["q_proj", "v_proj"],"mpt": ["Wqkv"]`. If you want to adapt to full modules on LLmMa model, please use `--lora_target_modules q_proj v_proj k_proj o_proj up_proj down_proj`, while it is `--lora_target_modules Wqkv out_proj up_proj down_proj` in MPT model.

- If you are using 4th Xeon or later (SPR etc.), please specify the `--bf16 --no_cuda` args, it will speedup the finetuning process without the loss of model's performance.;
- If you are using 3th Xeon or before (ICX etc.): please specify the `--no_cuda` args;
- If you are using GPU server: please specify the `--fp16` args.

### pure Lora Fine-tuning

For fine-tune with Lora only algorighm, you can try the following command.
```bash
# fine-tune with lora
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
    --algo "" \
    --trust_remote_code True \
    --no_cuda \
    2>&1 | tee $log_save_path
```

## Evaluate the model

For model evaluation, we follow the same method in [open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), which evaluate 4 key benchmarks in the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

- To setup the evaluate env, use the following command
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout b281b0921b636bc3
pip install -e .
```

- An example on evaluate on the hellaswag tasks
```python
python main.py --model hf-causal-experimental \
    --model_args "pretrained=/home/vmagent/app/data/llama-7b,use_accelerate=True" \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size auto --max_batch_size 16 \
    --output_path /home/vmagent/app/data/llm-eval/test

```
Note: 
- To view all the arguments that are supported by `--model_args`, please refer to [lm-evaluation-harness/HuggingFaceAutoLM](https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/models/huggingface.py#L60C7-L60C24).
- For example evaluation on more benchmarks, please refer to the [`evaluate_benchmark` folder](../../tests/deltatuner/evaluate_benchmark/).