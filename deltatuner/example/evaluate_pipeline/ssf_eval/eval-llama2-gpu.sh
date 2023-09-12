#llama2-ssf-qa
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True \
    --tasks truthfulqa_mc  --num_fewshot 0 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/llama2-7b-finetune-ssf-qa-gpu

#llama2-ssf-arc
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True \
    --tasks arc_challenge  --num_fewshot 25 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/llama2-7b-finetune-ssf-arc-gpu

#llama2-ssf-hellaswag
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True \
    --tasks hellaswag  --num_fewshot 10 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/llama2-7b-finetune-ssf-hellaswag-gpu

#llama2-ssf-mmlu
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True \
    --tasks hendrycksTest*  --num_fewshot 5 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/llama2-7b-finetune-ssf-merge-allmodules/llama2-7b-finetune-ssf-mmlu-gpu