#llama2-hellaswag
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/data/Llama-2-7b-hf,use_accelerate=True,dtype=bfloat16 \
    --device cpu \
    --tasks hellaswag  --num_fewshot 10 \
    --batch_size 32 --max_batch_size 32 

#llama2-ssf-hellaswag
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/data/Llama-2-7b-hf,peft=/home/data/llama2-7b-finetune-ssf,use_accelerate=True,dtype=bfloat16 \
    --device cpu \
    --tasks hellaswag  --num_fewshot 10 \
    --batch_size 32 --max_batch_size 32 \
    --output_path /home/data/llama2-7b-finetune-ssf/llama2-7b-finetune-ssf-hellaswag

#llama2-ssf-arc
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/data/Llama-2-7b-hf,peft=/home/data/llama2-7b-finetune-ssf,use_accelerate=True,dtype=bfloat16 \
    --device cpu \
    --tasks arc_challenge  --num_fewshot 25 \
    --batch_size 32 --max_batch_size 32 \
    --output_path /home/data/llama2-7b-finetune-ssf/llama2-7b-finetune-ssf-arc

#llama2-ssf-qa
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/data/Llama-2-7b-hf,peft=/home/data/llama2-7b-finetune-ssf,use_accelerate=True,dtype=bfloat16 \
    --device cpu \
    --tasks truthfulqa_mc  --num_fewshot 0 \
    --batch_size 32 --max_batch_size 32 \
    --output_path /home/data/llama2-7b-finetune-ssf/llama2-7b-finetune-ssf-truthqa

#llama2-ssf-mmlu
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/data/Llama-2-7b-hf,peft=/home/data/llama2-7b-finetune-ssf,use_accelerate=True,dtype=bfloat16 \
    --device cpu \
    --tasks hendrycksTest*  --num_fewshot 5 \
    --batch_size 32 --max_batch_size 32 \
    --output_path /home/data/llama2-7b-finetune-ssf/llama2-7b-finetune-ssf-mmlu