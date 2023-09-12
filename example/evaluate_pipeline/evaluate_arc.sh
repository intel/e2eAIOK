#!/bin/bash
set -x

cd /home/vmagent/app/lm-evaluation-harness

# evaluate llama2-7b-lora
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=/home/vmagent/app/data/Llama-2-7b-hf,peft=/home/vmagent/app/data/lora/llama2_lora_finetuned_model,use_accelerate=True \
	--tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
	--output_path /home/vmagent/app/data/llm-eval/llama2-7b-lora-arc_challenge

# evaluate llama2-7b-ssf
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/data/Llama-2-7b-hf,peft=/home/vmagent/app/data/llama2-7b-finetune-ssf,use_accelerate=True \
    --tasks arc_challenge  --num_fewshot 25 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/data/llm-eval/llama2-7b-finetune-ssf-arc_challenge

# evaluate mpt-7b-delta
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=/home/vmagent/app/data/mpt-7b,peft=/home/vmagent/app/data/lora/mpt_delta_tuned_model,use_accelerate=True,trust_remote_code=True,dtype=float16 \
	--tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
	--output_path /home/vmagent/app/data/llm-eval/mpt-7b-delta-arc_challenge

# evaluate mpt-7b-ssf
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/data/mpt-7b,peft=/home/vmagent/app/data/mpt-7b-finetune-ssf/,use_accelerate=True,trust_remote_code=True,tokenizer=/home/vmagent/app/data/gpt-neox-20b \
    --tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/data/llm-eval/mpt-7b-ssf-arc_challenge

# evaluate llama-7b-lora
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=/home/vmagent/app/data/llama-7b,peft=/home/vmagent/app/data/lora/llama_lora_finetuned_model,use_accelerate=True \
	--tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
	--output_path /home/vmagent/app/data/llm-eval/llama-7b-lora-arc_challenge

# evaluate mpt-7b-lora
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=/home/vmagent/app/data/mpt-7b,peft=/home/vmagent/app/data/lora/mpt_lora_finetuned_model,use_accelerate=True,trust_remote_code=True,dtype=float16 \
	--tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
	--output_path /home/vmagent/app/data/llm-eval/mpt-7b-lora-arc_challenge

