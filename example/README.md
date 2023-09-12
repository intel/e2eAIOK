DeltaTuner Fine-tuning
============

This example demonstrates how to finetune the pretrained large language model (LLM) with the instruction-following dataset by using the DeltaTuner and some foundamental models. Giving Fine-tuned model the textual instruction, it will respond with the textual response. This example have been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids.

# Prerequisite​

## 1. Environment​
- build the docker image
```shell
# in internal server
docker build --build-arg https_proxy=http://child-prc.intel.com:913 --build-arg http_proxy=http://child-prc.intel.com:913 -f docker/Dockerfile -t chatbot_finetune .
# in external server
docker build -f docker/Dockerfile -t chatbot_finetune .
```
- create docker container
```shell
# in internal server
docker run -it --name chatbot \
        --privileged --network host --ipc=host \
        --device=/dev/dri \
        -v /dev/shm:/dev/shm \
        -e http_proxy=http://child-prc.intel.com:913 \
        -e https_proxy=http://child-prc.intel.com:913 \
        -v /mnt/DP_disk1/dataset:/home/vmagent/app/data \
        -v /mnt/DP_disk1/yu:/home/vmagent/app \
        -w /home/vmagent/app/  \
        chatbot_finetune:latest \
        /bin/bash
```
```shell
# in external server
docker run -it --name chatbot \
        --privileged --network host --ipc=host \
        --device=/dev/dri \
        -v /dev/shm:/dev/shm \
        -v ~:/home/vmagent/app \
        -w /home/vmagent/app  \
        chatbot_finetune:latest \
        /bin/bash
```
```shell
# in internal gpu server
docker run -it --name chatbot \
        --privileged --network host --ipc=host \
        --device=/dev/dri \
        --runtime=nvidia \
        -v /dev/shm:/dev/shm \
        -e http_proxy=http://child-prc.intel.com:913 \
        -e https_proxy=http://child-prc.intel.com:913 \
        -v /mnt/DP_disk1/dataset:/home/vmagent/app/data \
        -v /mnt/DP_disk1/yu:/home/vmagent/app \
        -w /home/vmagent/app/  \
        chatbot_finetune:latest \
        /bin/bash
```
- Or you can direct create from the bare mental env
```shell
# both internal and external server
pip install -r requirements.txt
```

- Install the `deltatuner` python package

please clone this repo, then run:
`pip install -e .`
then you will be able to `import deltatuner` at any place in you code.

## 2. Prepare the Model

- MPT: Download [the released model on Huggingface](https://huggingface.co/mosaicml/mpt-7b)
- Llama-7B: Download [the released model on Huggingface](https://huggingface.co/huggyllama/llama-7b)
- Llama-2-7B: Download [the released model on Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)

## 3. Prepare Dataset
The [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). You can also download from [the released dataset on Huggingface](https://huggingface.co/datasets/tatsu-lab/alpaca), or you can download [the cleaned version of Alpaca on Huggingface](https://huggingface.co/datasets/yahma/alpaca-cleaned).


# Finetune

We employ the [LoRA approach](https://arxiv.org/pdf/2106.09685.pdf) to finetune the LLM efficiently.

## 1. Single Node Fine-tuning in Xeon SPR

### Lora Fine-tuning

For [MPT](https://huggingface.co/mosaicml/mpt-7b), use the below command line for finetuning on the Alpaca dataset. Only LORA supports MPT in PEFT perspective.it uses gpt-neox-20b tokenizer, so you need to define it in command line explicitly.This model also requires that trust_remote_code=True be passed to the from_pretrained method. This is because we use a custom MPT model architecture that is not yet part of the Hugging Face transformers package.

```bash
# in internal server
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/home/vmagent/app/data/mpt-7b" \
        --train_file "/home/vmagent/app/data/stanford_alpaca/alpaca_data.json" \
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
        --output_dir /home/vmagent/app/data/mpt_peft_finetuned_model \
        --peft lora \
        --trust_remote_code True \
        --no_cuda | tee mpt-lora-run-1epoch.log
```

```bash
# in external server
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/home/vmagent/app/dataset/mpt-7b" \
        --train_file "/home/vmagent/app/dataset/stanford_alpaca/alpaca_data.json" \
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
        --output_dir ./mpt_peft_finetuned_model \
        --peft lora \
        --trust_remote_code True \
        --no_cuda \
        --bf16 True | tee mpt-lora-run-1epoch.log
```

Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning on SPR, add `--bf16` argument will speedup the finetuning process without the loss of model's performance.

For model profile, add `--profile` argument, the forward time and backward time will print on the console.

For target modules of LoRA, use `--lora_target_modules` argument, the default target module of MPT and LLaMa model is `"llama": ["q_proj", "v_proj"],"mpt": ["Wqkv"]`. If you want to adapt to full modules on LLmMa model, please use `--lora_target_modules q_proj v_proj k_proj o_proj up_proj down_proj`, while it is `--lora_target_modules Wqkv out_proj up_proj down_proj` in MPT model.

### Fully Fine-tuning

For full fine-tune, you can try the following command.
```bash
# in external server
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/home/vmagent/app/dataset/mpt-7b" \
        --train_file "/home/vmagent/app/dataset/stanford_alpaca/alpaca_data.json" \
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
        --output_dir ./mpt_finetuned_model \
        --trust_remote_code True \
        --no_cuda \
        --bf16 True | tee mpt-full-finetune-run-1epoch.log
```
```bash
# in internal server
python instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "/home/vmagent/app/data/mpt-7b" \
        --train_file "/home/vmagent/app/data/stanford_alpaca/alpaca_data.json" \
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
        --output_dir /home/vmagent/app/data/mpt_finetuned_model \
        --trust_remote_code True \
        --no_cuda | tee mpt-full-finetune-run-1epoch.log
```

## 2. Multi-node Fine-tuning in Xeon SPR

We also supported Distributed Data Parallel finetuning on single node and multi-node settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, to finetune MPT through Distributed Data Parallel training, bash command will look like the following, where
- *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case,
- *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1,
- *`<NUM_NODES>`* is the number of nodes to use,
- *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.

> Also please note that to use CPU for training in each node with multi-node settings, argument `--no_cuda` is mandatory, and `--ddp_backend ccl` is required if to use ccl as the distributed backend. In multi-node setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

``` bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    finetune_clm.py \
        --model_name_or_path "mosaicml/mpt-7b" \
        --train_file "tatsu-lab/alpaca" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --do_eval \
        --validation_split_percentage 30 \
        --learning_rate 1e-4 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_steps 2000 \
        --save_total_limit 1 \
        --output_dir ./mpt_peft_finetuned_model \
        --peft lora \
        --no_cuda \
        --ddp_backend ccl \
        --bf16 True
```
If you have enabled passwordless SSH in cpu clusters, you could also use mpirun in master node to start the DDP finetune. Take llama alpaca finetune for example. follow the [hugginface guide](https://huggingface.co/docs/transformers/perf_train_cpu_many) to install Intel® oneCCL Bindings for PyTorch, IPEX

```shell
python -m pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu
```

oneccl_bindings_for_pytorch is installed along with the MPI tool set. Need to source the environment before using it.

for Intel® oneCCL >= 1.12.0
``` bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

for Intel® oneCCL whose version < 1.12.0
``` bash
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

The following command enables training with a total of 16 processes on 4 Xeons (node0/1/2/3, 2 sockets each node. taking node0 as the master node), ppn (processes per node) is set to 4, with two processes running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.

In node0, you need to create a configuration file which contains the IP addresses of each node (for example hostfile) and pass that configuration file path as an argument.
``` bash
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
 xxx.xxx.xxx.xxx #node2 ip
 xxx.xxx.xxx.xxx #node3 ip
```
Now, run the following command in node0 and **4DDP** will be enabled in node0 and node1 with BF16 auto mixed precision:
``` bash
export CCL_WORKER_COUNT=1
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip

## for DDP LORA for MPT
mpirun -f nodefile -n 16 -ppn 4 -genv OMP_NUM_THREADS=56 \
    python3 finetune_clm.py \
    --model_name_or_path mosaicml/mpt-7b \
    --train_file tatsu-lab/alpaca \
    --output_dir ./mpt_peft_finetuned_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4  \
    --logging_steps 100 \
    --peft lora \
    --group_by_length True \
    --dataset_concatenation \
    --do_train \
    --do_eval \
    --validation_split_percentage 30 \
    --trust_remote_code True \
    --no_cuda \
    --ddp_backend ccl \
    --bf16 True
```

## 3. Evaluate the model

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
# --device cuda:0/cpu
# --tasks hendrycksTest*
# --model_args dtype=bfloat16
python main.py --model hf-causal-experimental \
	--model_args "pretrained=/home/vmagent/app/data/llama-7b,use_accelerate=True" \
	--tasks hellaswag \
	--num_fewshot 10 \
	--batch_size auto --max_batch_size 16 \
	--output_path /home/vmagent/app/data/llm-eval/test

```
Note: to view all the arguments that are supported by `--model_args`, please refer to [lm-evaluation-harness/HuggingFaceAutoLM](https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/models/huggingface.py#L60C7-L60C24)

For example evaluation on more benchmarks, please refer to the folder [evaluate_pipeline](./evaluate_pipeline/).