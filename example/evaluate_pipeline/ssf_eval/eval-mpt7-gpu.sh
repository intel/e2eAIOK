#mpt7-ssf-arc
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True,tokenizer=/home/vmagent/app/LLM/data/gpt-neox-20b,dtype=float16 \
    --tasks arc_challenge  --num_fewshot 25 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/mpt-7b-finetune-ssf-merge-allmodules-arc-gpu

#mpt7-ssf-qa
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True,tokenizer=/home/vmagent/app/LLM/data/gpt-neox-20b,dtype=float16 \
    --tasks truthfulqa_mc  --num_fewshot 0 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/mpt-7b-finetune-ssf-merge-allmodules-qa-gpu

#mpt7-ssf-mmlu
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True,tokenizer=/home/vmagent/app/LLM/data/gpt-neox-20b,dtype=float16 \
    --tasks hendrycksTest*  --num_fewshot 5 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/mpt-7b-finetune-ssf-merge-allmodules-mmlu-gpu

#mpt7-ssf-hellaswag
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True,tokenizer=/home/vmagent/app/LLM/data/gpt-neox-20b,dtype=float16 \
    --tasks hellaswag  --num_fewshot 10 \
    --batch_size auto --max_batch_size 32 \
    --output_path /home/vmagent/app/LLM/data/mpt-7b-finetune-ssf-merge-allmodules/mpt-7b-finetune-ssf-merge-allmodules-hellaswag-gpu