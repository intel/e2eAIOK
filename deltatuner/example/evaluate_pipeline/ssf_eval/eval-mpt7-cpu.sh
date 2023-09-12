python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/home/data/mpt-7b-finetune-ssf-merge-allmodules/merged_model,use_accelerate=True,trust_remote_code=True,tokenizer=/home/data/gpt-neox-20b \
    --device cpu \
    --tasks hellaswag  --num_fewshot 10 \
    --batch_size 16 --max_batch_size 16 \
    --output_path /home/data/mpt-7b-finetune-ssf-merge-allmodules/mpt-7b-finetune-ssf-merge-allmodules-hellaswag