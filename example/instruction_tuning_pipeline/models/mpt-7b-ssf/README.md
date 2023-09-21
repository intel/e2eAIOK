---
license: apache-2.0
tags:
- Composer
- MosaicML
- llm-foundry
- StreamingDatasets
datasets:
- mc4
- c4
- togethercomputer/RedPajama-Data-1T
- bigcode/the-stack
- allenai/s2orc
inference: false
---

# MPT-7B

MPT-7B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code.
This model was trained by [MosaicML](https://www.mosaicml.com).

MPT-7B is part of the family of MosaicPretrainedTransformer (MPT) models, which use a modified transformer architecture optimized for efficient training and inference.

These architectural changes include performance-optimized layer implementations and the elimination of context length limits by replacing
positional embeddings with Attention with Linear Biases ([ALiBi](https://arxiv.org/abs/2108.12409)).
Thanks to these modifications, MPT models can be trained with high throughput efficiency and stable convergence.
MPT models can also be served efficiently with both standard HuggingFace pipelines and NVIDIA's [FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

This model uses the MosaicML LLM codebase, which can be found in the [llm-foundry repository](https://github.com/mosaicml/llm-foundry). It was trained by MosaicML’s NLP team on the [MosaicML platform](https://www.mosaicml.com/training) for LLM pretraining, finetuning, and inference.

### How is this model different?

MPT-7B is

* **Licensed for the possibility of commercial use** (unlike [LLaMA](https://arxiv.org/abs/2302.13971)).
* **Trained on a large amount of data** (1T tokens like [LLaMA](https://arxiv.org/abs/2302.13971) vs. 300B for [Pythia](https://github.com/EleutherAI/pythia), 300B for [OpenLLaMA](https://github.com/openlm-research/open_llama), and 800B for [StableLM](https://github.com/Stability-AI/StableLM)).
* **Prepared to handle extremely long inputs** thanks to [ALiBi](https://arxiv.org/abs/2108.12409) (we finetuned [MPT-7B-StoryWriter-65k+](https://huggingface.co/mosaicml/mpt-7b-storywriter) on up to 65k inputs and can handle up to 84k vs. 2k-4k for other open source models).
* **Capable of fast training and inference** (via [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer))
* **Equipped with highly efficient open-source training code** via the [llm-foundry repository](https://github.com/mosaicml/llm-foundry)

### Models finetuned off MPT-7B:

The following models are finetuned on MPT-7B:

* [MPT-7B-StoryWriter-65k+](https://huggingface.co/mosaicml/mpt-7b-storywriter): a model designed to read and write fictional stories with super long context lengths.
Built by finetuning MPT-7B with a context length of 65k tokens on a filtered fiction subset of the [books3 dataset](https://huggingface.co/datasets/the_pile_books3).
At inference time, thanks to [ALiBi](https://arxiv.org/abs/2108.12409), MPT-7B-StoryWriter-65k+ can extrapolate even beyond 65k tokens.
We demonstrate generations as long as 80k tokens on a single A100-80GB GPU in our [blogpost](www.mosaicml.com/blog/mpt-7b).
  * License: Apache 2.0

* [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct): a model for short-form instruction following.
Built by finetuning MPT-7B on a [dataset](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) we also release, derived from the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and the [Anthropic Helpful and Harmless (HH-RLHF)](https://huggingface.co/datasets/Anthropic/hh-rlhf) datasets.
  * License: _CC-By-SA-3.0_
  * [Demo on Hugging Face Spaces](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct)

* [MPT-7B-Chat](https://huggingface.co/mosaicml/mpt-7b-chat): a chatbot-like model for dialogue generation.
Built by finetuning MPT-7B on the [ShareGPT-Vicuna](https://huggingface.co/datasets/jeffwan/sharegpt_vicuna), [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3),
 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), and [Evol-Instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k) datasets.
  * License: _CC-By-NC-SA-4.0_
  * [Demo on Hugging Face Spaces](https://huggingface.co/spaces/mosaicml/mpt-7b-chat)

## Model Date

May 5, 2023

## Model License

Apache-2.0

## Documentation

* [Blog post: Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.mosaicml.com/blog/mpt-7b)
* [Codebase (mosaicml/llm-foundry repo)](https://github.com/mosaicml/llm-foundry/)
* Questions: Feel free to contact us via the [MosaicML Community Slack](https://mosaicml.me/slack)!


## How to Use

This model is best used with the MosaicML [llm-foundry repository](https://github.com/mosaicml/llm-foundry) for training and finetuning.

```python
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b',
  trust_remote_code=True
)
```
Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method.
This is because we use a custom `MPT` model architecture that is not yet part of the Hugging Face `transformers` package.
`MPT` includes options for many training efficiency features such as [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi](https://arxiv.org/abs/2108.12409), [QK LayerNorm](https://arxiv.org/abs/2010.04245), and more.

To use the optimized [triton implementation](https://github.com/openai/triton) of FlashAttention, you can load the model on GPU (`cuda:0`) with `attn_impl='triton'` and with `bfloat16` precision:
```python
import torch
import transformers

name = 'mosaicml/mpt-7b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)
```

Although the model was trained with a sequence length of 2048, ALiBi enables users to increase the maximum sequence length during finetuning and/or inference. For example:

```python
import transformers

name = 'mosaicml/mpt-7b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True
)
```

This model was trained with the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
```

The model can then be used, for example, within a text-generation pipeline.  
Note: when running Torch modules in lower precision, it is best practice to use the [torch.autocast context manager](https://pytorch.org/docs/stable/amp.html).

```python
from transformers import pipeline

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

with torch.autocast('cuda', dtype=torch.bfloat16):
    print(
        pipe('Here is a recipe for vegan banana bread:\n',
            max_new_tokens=100,
            do_sample=True,
            use_cache=True))
```

## Model Description

The architecture is a modification of a standard decoder-only transformer.

The model has been modified from a standard transformer in the following ways:
* It uses [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf)
* It uses [ALiBi (Attention with Linear Biases)](https://arxiv.org/abs/2108.12409) and does not use positional embeddings
* It does not use biases


| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 6.7B |
|n_layers | 32 |
| n_heads | 32 |
| d_model | 4096 |
| vocab size | 50432 |
| sequence length | 2048 |



## Training Data

### Streaming Datasets

Data was formatted using the MosaicML [StreamingDataset](https://github.com/mosaicml/streaming) library to host our data in object storage and efficiently stream it to our compute cluster during training.
StreamingDataset obviates the need to download the whole dataset before starting training, and allows instant resumption of training from any point in the dataset.


### Data Mix

The model was trained for 1T tokens (with batch size 1760 and sequence length 2048). It was trained on the following data mix:


| Data Source | Number of Tokens in Source | Proportion | Effective Number of Tokens | Epochs |
|-------------|----------------------------|------------|----------------------------|--------|
| mC4 3.1.0 - English | 417.99 B | 0.33 | 330 B | 0.14 |
| C4 - English - SemDedup 80% | 100.42 B | 0.299 | 299 B | 2.98 |
| RedPajama - CommonCrawl | 878.45 B | 0.1 | 100 B | 0.11 |
| The Stack - Selected Languages | 463.78 B | 0.1 | 100 B | 0.22 |
| RedPajama - Wikipedia - En | 4.87 B | 0.04 | 40 B | 8.21 |
| The Stack - Markdown | 107.07 B | 0.035 | 35 B | 0.33 |
| S2ORC | 48.85 B | 0.033 | 33 B | 0.68 |
| RedPajama - Books | 26.02 B | 0.03 | 30B | 1.15 |
| RedPajama - arXiv | 28.10 B | 0.019 | 19 B | 0.68 |
| RedPajama - StackExchange | 20.54 B | 0.014 | 14 B |0.68 |

Samples for each batch were selected from one of the datasets with the probability specified above.
The examples were shuffled within each dataset, and each example was constructed from as many sequences from that dataset as were necessary to fill the 2048 sequence length.

The data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer. This BPE tokenizer has a number of desirable characteristics,
most of which are relevant for tokenizing code:
(1) It was trained on a diverse mix of data that includes code (The Pile)
(2) It applies consistent space delimitation, unlike the GPT2 tokenizer which tokenizes inconsistently depending on the presence of prefix spaces
(3) It contains tokens for repeated space characters, which allows superior compression of text with large amounts of repeated space characters.

The model vocabulary size of 50432 was set to be a multiple of 128 (as in [MEGATRON-LM](https://arxiv.org/abs/1909.08053)), model flop utilization (MFU) increased by up to four percentage points.

### Training Configuration

This model was trained on 440 A100-40GBs for about 9.5 days using the [MosaicML Platform](https://www.mosaicml.com/platform).
The model was trained with sharded data parallelism using [FSDP](https://pytorch.org/docs/stable/fsdp.html) and used the [LION](https://arxiv.org/abs/2302.06675) optimizer.

## Limitations and Biases

_The following language is modified from [EleutherAI's GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)_

MPT-7B (Base) is **not** intended for deployment without finetuning.
It should not be used for human-facing interactions without further guardrails and user consent.

MPT-7B can produce factually incorrect output, and should not be relied on to produce factually accurate information.
MPT-7B was trained on various public datasets.
While great efforts have been taken to clean the pretraining data, it is possible that this model could generate lewd, biased or otherwise offensive outputs.


## MosaicML Platform

If you're interested in [training](https://www.mosaicml.com/training) and [deploying](https://www.mosaicml.com/inference) your own MPT or LLMs on the MosaicML Platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=huggingface&utm_medium=referral&utm_campaign=mpt-7b).

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please cosult an attorney before using this model for commercial purposes.

## Citation

Please cite this model using the following format:

```
@online{MosaicML2023Introducing,
    author    = {MosaicML NLP Team},
    title     = {Introducing MPT-7B: A New Standard for Open-Source,
    Commercially Usable LLMs},
    year      = {2023},
    url       = {www.mosaicml.com/blog/mpt-7b},
    note      = {Accessed: 2023-05-05},
    urldate   = {2023-05-05}
}
```
