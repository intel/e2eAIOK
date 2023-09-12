#!/bin/bash
set -x

cp -r ~/lm-evaluation-harness /home/vmagent/app/lm-evaluation-harness

cd /home/vmagent/app/lm-evaluation-harness

# evaluate llama-2-7b-lora on hellaswag
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/home/vmagent/app/data/lora/llama2_lora_finetuned_model,use_accelerate=True \
	--tasks hellaswag  --num_fewshot 10 \
	--batch_size auto --max_batch_size 32 \
	--output_path /home/vmagent/app/data/llm-eval/llama2-7b-lora-hellaswag

# evaluate llama-2-7b on hellaswag
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=meta-llama/Llama-2-7b-hf,use_accelerate=True \
	--tasks hellaswag  --num_fewshot 10 \
	--batch_size auto --max_batch_size 32 \
	--output_path /home/vmagent/app/data/llm-eval/llama2-7b-hellaswag