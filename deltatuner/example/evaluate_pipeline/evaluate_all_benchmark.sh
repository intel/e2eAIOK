#!/bin/bash
set -x

# meta-llama/Llama-2-7b-hf, mosaicml/mpt-7b
pretrained=/home/vmagent/app/data/Llama-2-7b-hf
peft=/home/vmagent/app/data/lora/llama2_delta_tuned_model
output_path_prefix=/home/vmagent/app/data/llm-eval/llama2-7b-delta
# '', ',trust_remote_code=True,dtype=float16'
extra_model_args=',trust_remote_code=True,dtype=float16'


cd /home/vmagent/app/lm-evaluation-harness

# evaluate on hellaswag
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=$pretrained,peft=$peft,use_accelerate=True$extra_model_args \
	--tasks hellaswag  --num_fewshot 10 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-hellaswag

# evaluate on mmlu
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=$pretrained,peft=$peft,use_accelerate=True$extra_model_args \
	--tasks hendrycksTest*  --num_fewshot 5 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-mmlu

# evaluate on arc_challenge
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=$pretrained,peft=$peft,use_accelerate=True$extra_model_args \
	--tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-arc_challenge

# evaluate on truthfulqa_mc
python main.py \
    --model hf-causal-experimental \
	--model_args pretrained=$pretrained,peft=$peft,use_accelerate=True$extra_model_args \
	--tasks truthfulqa_mc  --num_fewshot 0 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-truthqa

cd /home/vmagent/app/dtuner/example/evaluate_pipeline
bash cal_avg_acc.sh $output_path_prefix-mmlu