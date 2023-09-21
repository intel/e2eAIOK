Deltatuner
============
Deltatuner is an extension for [Peft](https://github.com/huggingface/peft) to improve LLM fine-tuning speed through multiple optimizations, including: leverage the compact model constructor [DE-NAS](https://github.com/intel/e2eAIOK/tree/main/e2eAIOK/DeNas) to construct/modify the compact delta layers in a hardware-aware and train-free approach, and adding more new deltatuning algorithms.

![Architecure](./doc/deltatuner.png)

Below is an example showing how to optimize the finetuning process of [Intel NeuralChat](https://github.com/intel/intel-extension-for-transformers/tree/main/workflows/chatbot/fine_tuning).

# Installation
- install the python package
```shell
git clone https://github.com/intel-innersource/frameworks.bigdata.AIDK.git deltatuner
cd deltatuner
git checkout deltatuner
pip install -e .
```


# Fine-tuning Use Cases

We use delatuner to optimize the [LoRA approach](https://arxiv.org/pdf/2106.09685.pdf) to finetune the LLM efficiently. 

## Fine-tuning on MPT-7B
For [MPT](https://huggingface.co/mosaicml/mpt-7b), adding the following command to use the delatuner optimizations in the [LLM fine-tuning script](./example/instruction_tuning_pipeline/finetune_clm.py) for finetuning on the Alpaca dataset. 

```python
from delta import deltatuner, deltatuner_args
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from delta import deltatuner, deltatuner_args

# import model from huggingface
model_id =  "mosaicml/mpt-7b"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# adding the lora componenents with peft
config = LoraConfig()
lora_model = get_peft_model(model, config) 
# delatuner optimize the model with best lora layer configuration
deltatuning_args = deltatuner_args.DeltaTunerArguments()
deltatuner_model = deltatuner.optimize(model=lora_model, tokenizer=tokenizer, deltatuning_args=deltatuning_args)
...
```

## More examples on Fine-tuning other LLMs

Please refer to [example page](https://github.com/intel-innersource/frameworks.bigdata.AIDK/tree/deltatuner/example) for more use cases on fine-tuning other LLMs with the help of DeltaTuner.

# Model supported matrix

## Causal Language Modeling
### Causal Language Modeling
| Model        | LoRA | SSF  |
|--------------| ---- | ---- |
| GPT-2        |  |  |
| GPT-J        |  |  |
| Bloom        |  |  |
| OPT          |  |  |
| GPT-Neo      |  |  |
| Falcon       |  |  |
| Flan-T5      |  |  |
| LLaMA        | ✅  | ✅  |
| MPT          | ✅  | ✅  |
