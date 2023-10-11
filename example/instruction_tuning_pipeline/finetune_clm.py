#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
import logging
import os
import errno
import sys
import transformers
from transformers.modeling_utils import unwrap_model
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel
)
from peft.tuners.adaption_prompt import AdaptionPromptConfig
from deltatuner import deltatuner, SSFConfig, DeltaTunerModel, DeltaTunerArguments
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from typing import Optional, List, Union
import copy
import re
import torch
import importlib.util
from transformers.utils.import_utils import is_optimum_available

IGNORE_INDEX = -100


logger = logging.getLogger(__name__)


def is_optimum_habana_available():
    return is_optimum_available() and importlib.util.find_spec("optimum.habana") != None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "should enable when using custom model architecture that is not yet part of the Hugging Face transformers package like MPT)."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep in memory the loaded dataset. Defaults to False."
        },
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load processed dataset from this path."
        },
    )
    dataset_concatenation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to concatenate the sentence for more efficient training."
        },
    )


@dataclass
class FinetuneArguments:
    """
    Arguments of finetune we are going to apply on the model.
    """

    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank parameter in the LoRA method."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    adapter_layers: int = field(
        default=30,
        metadata={"help": "adapter layer number in the LLaMA-adapter."},
    )
    adapter_len: int = field(
        default=10,
        metadata={
            "help": "The length of the adaption prompt to insert in the LLaMA-adapter."
        },
    )
    num_virtual_tokens: int = field(
        default=10,
        metadata={
            "help": "The length of the vitrual tokens to insert in P-tuning/Prompt-tuning/Prefix-tuning"
        },
    )
    ptun_hidden_size: int = field(
        default=1024,
        metadata={"help": "The encoder hidden size in P-tuning"},
    )
    peft: Optional[str] = field(
        default="",
        metadata={
            "help": ("apply peft. default set to lora"),
            "choices": ["llama_adapter", "lora", "ptun", "prefix", "prompt", ""],
        },
    )
    resume_peft: Optional[str] = field(
        default="",
        metadata={"help": "the path of peft model to resume"},
    )
    delta: Optional[str] = field(
        default=None,
        metadata={
            "help": ("whether to use the deltatuner optimization"),
            "choices": ["auto", "lora", "ssf"]
        },
    )
    profile: bool = field(
        default=False,
        metadata={"help": "Profile the model to get the forward and backward time"}
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    habana: bool = field(
        default=False,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    debugs: bool = field(
        default=False,
        metadata={"help": "for debug"},
    )
    save_merged_model: bool = field(
        default=False,
        metadata={"help": "save merged model"},
    )
    merge_model_code_dir: Optional[str] = field(
        default="",
        metadata={"help": "the code path of base model with enable bias on target modules for ssf algo"},
    )

PROMPT_DICT = {
    "prompt_with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def create_prompts(examples):
    prompts = {}
    prompts["source"] = []
    prompts["target"] = []
    for example in examples:
        prompt_template = (
            PROMPT_DICT["prompt_with_input"]
            if example["input"] != ""
            else PROMPT_DICT["prompt_without_input"]
        )
        source = prompt_template.format_map(example)
        prompts["source"].append(source)
        prompts["target"].append(example["output"])
    return prompts


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    if not is_optimum_habana_available():
        parser = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, FinetuneArguments, DeltaTunerArguments)
        )
    else:
        from optimum.habana import GaudiTrainingArguments

        parser = HfArgumentParser(
            (ModelArguments, DataArguments, GaudiTrainingArguments, FinetuneArguments, DeltaTunerArguments)
        )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, finetune_args, deltatuner_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            finetune_args,
            deltatuner_args
        ) = parser.parse_args_into_dataclasses()

    print(f'finetune_args is \n {finetune_args}')

    if finetune_args.habana:
        if not is_optimum_habana_available():
            raise ImportError(
                "optimum habana is not installed. refer https://github.com/huggingface/optimum-habana"
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary
    b16 = training_args.fp16 or training_args.bf16
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f"\ndistributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {b16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True if model_args.trust_remote_code else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        raise ValueError("Please provide value for model_name_or_path or config_name.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    elif config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )

        if "validation" not in raw_datasets.keys() and training_args.do_eval:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys() and training_args.do_eval:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # Preprocessing the datasets.
    for key in raw_datasets:
        prompts = create_prompts(raw_datasets[key])
        columns_to_be_removed = list(raw_datasets[key].features.keys())
        raw_datasets[key] = raw_datasets[key].add_column(
            "prompt_sources", prompts["source"]
        )
        raw_datasets[key] = raw_datasets[key].add_column(
            "prompt_targets", prompts["target"]
        )
        raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)

    # Load model
    if model_args.model_name_or_path:
        model_dtype = torch.bfloat16 if training_args.bf16 else None
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=True if model_args.trust_remote_code else None,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError(
            "Must provide model_name_or_path to load a pretrained CausalLM model."
        )

    if re.search("llama", model.config.architectures[0], re.IGNORECASE):
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2

    if (
        hasattr(model.generation_config, "pad_token_id")
        and model.generation_config.pad_token_id is not None
    ):
        tokenizer.pad_token_id = model.generation_config.pad_token_id
    if (
        hasattr(model.generation_config, "eos_token_id")
        and model.generation_config.eos_token_id is not None
    ):
        tokenizer.eos_token_id = model.generation_config.eos_token_id
    if (
        hasattr(model.generation_config, "bos_token_id")
        and model.generation_config.bos_token_id is not None
    ):
        tokenizer.bos_token_id = model.generation_config.bos_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        results = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        for i in range(len(results["input_ids"])):
            if (
                results["input_ids"][i][-1] != tokenizer.eos_token_id
                and len(results["input_ids"][i]) < data_args.max_seq_length
                and add_eos_token
            ):
                results["input_ids"][i].append(tokenizer.eos_token_id)
                results["attention_mask"][i].append(1)

        results["labels"] = copy.deepcopy(results["input_ids"])
        results["input_id_len"] = [len(result) for result in results["input_ids"]]
        return results

    def preprocess_function(examples):
        st = [
            s + t
            for s, t in zip(examples["prompt_sources"], examples["prompt_targets"])
        ]
        examples_tokenized = tokenize(st)
        input_ids = examples_tokenized["input_ids"]
        labels = examples_tokenized["labels"]
        if not finetune_args.train_on_inputs:
            sources_tokenized = tokenize(
                examples["prompt_sources"], add_eos_token=False
            )
            for label, source_len in zip(labels, sources_tokenized["input_id_len"]):
                label[:source_len] = [IGNORE_INDEX] * source_len
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=examples_tokenized["attention_mask"],
        )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if data_args.dataset_concatenation:

        def concatenate_data(dataset, max_seq_length):
            concatenated_dataset = {}
            for column in dataset.features:
                concatenated_data = [
                    item for sample in dataset[column] for item in sample
                ]
                reshaped_data = [
                    concatenated_data[i * max_seq_length : (i + 1) * max_seq_length]
                    for i in range(len(concatenated_data) // max_seq_length)
                ]
                concatenated_dataset[column] = reshaped_data
            return datasets.Dataset.from_dict(concatenated_dataset)

        tokenized_datasets_ = tokenized_datasets["train"].remove_columns(
            ["prompt_sources", "prompt_targets"]
        )
        tokenized_datasets["train"] = concatenate_data(
            tokenized_datasets_, data_args.max_seq_length
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    logger.info(
        "Using data collator of type {}".format(data_collator.__class__.__name__)
    )

    if finetune_args.peft:
        # PEFT settings
        if finetune_args.peft == "lora":
            peft_config = LoraConfig(
                r=finetune_args.lora_rank,
                lora_alpha=finetune_args.lora_alpha,
                lora_dropout=finetune_args.lora_dropout,
                target_modules=finetune_args.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        elif finetune_args.peft == "llama_adapter":
            peft_config = AdaptionPromptConfig(
                adapter_layers=finetune_args.adapter_layers,
                adapter_len=finetune_args.adapter_len,
                task_type="CAUSAL_LM",
            )
        elif finetune_args.peft == "ptun":
            peft_config = PromptEncoderConfig(
                num_virtual_tokens=finetune_args.num_virtual_tokens,
                encoder_hidden_size=finetune_args.ptun_hidden_size,
                task_type="CAUSAL_LM",
            )
        elif finetune_args.peft == "prefix":
            peft_config = PrefixTuningConfig(
                num_virtual_tokens=finetune_args.num_virtual_tokens,
                task_type="CAUSAL_LM",
            )
        elif finetune_args.peft == "prompt":
            peft_config = PromptTuningConfig(
                num_virtual_tokens=finetune_args.num_virtual_tokens,
                task_type="CAUSAL_LM",
            )

        if finetune_args.resume_peft != "":
            model = PeftModel.from_pretrained(model, finetune_args.resume_peft, config=peft_config)
        else:
            model = get_peft_model(model, peft_config)
        logger.info("***original optimized model parameter***")
        model.print_trainable_parameters()

    if finetune_args.delta:
        if finetune_args.resume_peft != "":
            model = DeltaTunerModel.from_pretrained(model, finetune_args.resume_peft, denas_config=deltatuner_args)
        else:
            model = deltatuner.optimize(model, tokenizer, algo=finetune_args.delta, deltatuning_args=deltatuner_args)
        logger.info("***deltatuner optimized model parameter***")
        model.print_trainable_parameters()
         
    if finetune_args.debugs:
        if training_args.do_train:
            train_dataset = train_dataset.select(range(8))
        if training_args.do_eval:
            eval_dataset = eval_dataset.select(range(8))

    if not finetune_args.habana:
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        from optimum.habana import GaudiConfig, GaudiTrainer

        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True
        # Initialize our Trainer
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    if finetune_args.profile:
        profile_model(model, train_dataset, data_collator, training_args)

    if training_args.do_train:
        logger.info("*** Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        with training_args.main_process_first(desc="save model"):
            if is_main_process(training_args.local_rank):
                unwrapped_model = unwrap_model(model)
                unwrapped_model.save_pretrained(
                    training_args.output_dir, state_dict=unwrapped_model.state_dict()
                )

    if finetune_args.save_merged_model:
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            saved_dir = os.path.join(training_args.output_dir, "merged_model")
            os.makedirs(saved_dir, exist_ok=True)
            print(f"copy base model config to {saved_dir}")
            os.system(f"cp {model_args.model_name_or_path}/* {saved_dir}")
            print(f"Save merged model to {saved_dir}")
            torch.save(model.state_dict(), os.path.join(saved_dir, "pytorch_model.bin"))
            if finetune_args.delta == 'ssf':
                if os.path.exists(finetune_args.merge_model_code_dir):
                    print(f"copy merged model code to {saved_dir}")
                    os.system(f"cp {finetune_args.merge_model_code_dir}/* {saved_dir}")
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), finetune_args.merge_model_code_dir)
                pre_train_config_file = os.path.join(saved_dir, "config.json")
                with open(pre_train_config_file, "r") as file:
                    config_json = json.load(file)
                adapter_config_file = os.path.join(finetune_args.resume_peft, "adapter_config.json")
                with open(adapter_config_file, "r") as file:
                    adapter_config_json = json.load(file)
                config_json['target_modules'] = adapter_config_json['target_modules']

                best_structure_file = os.path.join(finetune_args.resume_peft, "best_model_structure.txt")
                if os.path.isfile(best_structure_file):
                    with open(best_structure_file, "r") as file:
                        best_structure_json = json.loads(file.readline().strip())
                    config_json['best_model_structure'] = best_structure_json
            
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        
        total_token = sum(eval_dataset.map(lambda example: {"sentence_len": len(example["input_ids"])})["sentence_len"][:])
        metrics["eval_tokens"] = total_token

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def profile_model(model, train_dataset, data_collator, args):
    from transformers import get_scheduler
    from torch.utils.data import DataLoader

    saving_dir_name = 'hf-training-torch/tmp'

    model.train()
    train_dataset = train_dataset.select(range(100))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=data_collator
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_epochs * len(train_dataloader),
    )

    # torch.profiler.ProfilerActivity.CUDA
    # profile_memory=True,
    # schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=2),
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], 
                            on_trace_ready=torch.profiler.tensorboard_trace_handler(saving_dir_name),
                            schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=3, active=7, repeat=1),
    ) as prof:
        with torch.profiler.record_function("model_all"):

            for batch in train_dataloader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with torch.profiler.record_function("model_inference"):
                    outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                prof.step()
    logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    sys.exit(0)


if __name__ == "__main__":
    main()
