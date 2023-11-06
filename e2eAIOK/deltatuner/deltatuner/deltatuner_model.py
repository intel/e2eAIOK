import os
import torch
import json
import inspect
import random
import logging
import numpy as np
from typing import Any, Optional, Union
from peft.utils import (
    PeftType,
    get_peft_model_state_dict, 
    set_peft_model_state_dict,
    WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME
)

from transformers import PreTrainedModel, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, AdaLoraConfig
from peft.utils import PeftType, PromptLearningConfig, WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME, hub_file_exists, set_peft_model_state_dict
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate.hooks import AlignDevicesHook, remove_hook_from_submodules, add_hook_to_module
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from .deltatuner_args import DeltaTunerArguments
from .tuner import DeltaLoraModel, DeltaLoraSearchSpace, DeltaSSFModel, DeltaSSFSearchSpace
from .search import SearchEngineFactory, Timer
from .search.utils import network_latency
from .utils import DeltaTunerType, get_deltatuner_model_state_dict, set_deltatuner_model_state_dict, BEST_MODEL_STRUCTURE_DEFAULT_NAME
from typing import Any, Dict, List, Optional, Union

DELTATUNNER_TO_MODEL_MAPPING = {
     PeftType.LORA: DeltaLoraModel,
     DeltaTunerType.SSF: DeltaSSFModel,
}

DELTATUNNER_TO_SEARCH_SPACE = {
     PeftType.LORA: DeltaLoraSearchSpace,
     DeltaTunerType.SSF: DeltaSSFSearchSpace
}

MODEL_TYPE_TO_LAYER_NAME = {
    "mpt": "n_layers"
}

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DeltaTunerModel(PeftModel, torch.nn.Module):
    def __init__(self, model, peft_config: PeftConfig, adapter_name: str = "default", denas_config: DeltaTunerArguments = None, tokenizer: AutoTokenizer = None):
        torch.nn.Module.__init__(self)
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.best_model_param = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.peft_config[adapter_name] = peft_config
        logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('deltatuner')
        if peft_config.peft_type in (DeltaTunerType.SSF):
            self.base_model = DELTATUNNER_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)

            if getattr(model, "is_gradient_checkpointing", True):
                model = self._prepare_model_for_gradient_checkpointing(model)

            # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
            # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
            # behavior we disable that in this line.
            if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
                self.base_model.config.pretraining_tp = 1

        self.denas_config = denas_config
        if self.denas_config.denas:
            self._init_denas_params_()
            search_space, search_space_name = DELTATUNNER_TO_SEARCH_SPACE[peft_config.peft_type].generate_search_space(model, self.denas_config)
            supernet_config =  DELTATUNNER_TO_SEARCH_SPACE[peft_config.peft_type].generate_supernet(model, self.denas_config, search_space)
            if peft_config.peft_type in (DeltaTunerType.SSF):
                super_net = self.base_model
            else:
                super_net = DELTATUNNER_TO_MODEL_MAPPING[peft_config.peft_type](model, self.peft_config, adapter_name, supernet_config)
            if self.denas_config.best_model_structure and os.path.exists(self.denas_config.best_model_structure):
                with open(self.denas_config.best_model_structure, "r+") as best_model_param_file:
                    self.best_model_param = json.loads(best_model_param_file.readline().strip())
            else:
                self.denas_config.search_space_name = search_space_name
                self.best_model_param = json.loads(self.search(self.denas_config, super_net, search_space))
            self.base_model = super_net.set_sample_config(self.best_model_param)

    def _init_denas_params_(self):
        if self.config.model_type in MODEL_TYPE_TO_LAYER_NAME:
            self.denas_config.layer_name = MODEL_TYPE_TO_LAYER_NAME[self.config.model_type]
        self.denas_config.model_id = self.base_model.config._name_or_path
        self.denas_config.tokenizer = self.tokenizer
        self.denas_config.max_param_limits = sum(param.numel() for param in self.base_model.parameters() if param.requires_grad) / 10.**6 if self.denas_config.max_param_limits is None else self.denas_config.max_param_limits
        if self.tokenizer:
            self.denas_config.budget_latency_max = network_latency(self.base_model, self.tokenizer, batch_size=self.denas_config.batch_size) if self.denas_config.budget_latency_max is not None else self.denas_config.budget_latency_max

    def search(self, denas_config, super_net, search_space):
        setup_seed(denas_config.random_seed)
        with Timer("DE-NAS search best model"):
            searcher = SearchEngineFactory.create_search_engine(params=denas_config, super_net=super_net, search_space=search_space, peft_type=self.peft_type)
            best_structure=searcher.search()
        self.logger.info(f"DE-NAS completed, best structure is {best_structure}")
        return best_structure

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            if peft_config.peft_type in (DeltaTunerType.SSF):
                output_state_dict = get_deltatuner_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            else:
                output_state_dict = get_peft_model_state_dict(
                    self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
                )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if safe_serialization:
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=isinstance(peft_config, PromptLearningConfig)
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

        if self.best_model_param is not None:
            with open(os.path.join(save_directory, BEST_MODEL_STRUCTURE_DEFAULT_NAME), "w+") as fout:
                fout.write(json.dumps(self.best_model_param))

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
        from .mapping import DELTATUNER_TYPE_TO_CONFIG_MAPPING, MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING
        denas_config = cls._get_denas_config(model_id, **kwargs)

        # load the config
        if config is None:
            peft_type_name = PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                )
            if peft_type_name in ['SSF']:
                config = DELTATUNER_TYPE_TO_CONFIG_MAPPING[peft_type_name].from_pretrained(model_id, **kwargs)
            else:
                config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type_name].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name, denas_config)
        else:
            model = MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING[config.task_type](model, config, adapter_name, denas_config)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        return model

    @classmethod
    def _get_denas_config(cls, model_id: str, **kwargs: Any):
        hf_hub_download_kwargs, kwargs = cls._split_kwargs(kwargs)

        # load weights if any
        path = (
            os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else model_id
        )

        if os.path.exists(os.path.join(path, BEST_MODEL_STRUCTURE_DEFAULT_NAME)):
            filename = os.path.join(path, BEST_MODEL_STRUCTURE_DEFAULT_NAME)
        else:
            has_remote_structure_file = hub_file_exists(
                model_id,
                BEST_MODEL_STRUCTURE_DEFAULT_NAME,
                revision=hf_hub_download_kwargs.get("revision", None),
                repo_type=hf_hub_download_kwargs.get("repo_type", None),
            )

            if has_remote_structure_file:
                filename = hf_hub_download(
                    model_id,
                    BEST_MODEL_STRUCTURE_DEFAULT_NAME,
                    **hf_hub_download_kwargs,
                )
            else:
                raise ValueError(
                    f"Can't find structure for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {BEST_MODEL_STRUCTURE_DEFAULT_NAME} is present at {model_id}."
                )

        denas_config = DeltaTunerArguments()
        denas_config.denas = True
        denas_config.best_model_structure = filename
        
        return denas_config


    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs: Any):
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
        from .mapping import DELTATUNER_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)

        if adapter_name not in self.peft_config:
            # load the config
            peft_type_name = PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            if peft_type_name in ["SSF"]:
                peft_config = DELTATUNER_TYPE_TO_CONFIG_MAPPING[peft_type_name].from_pretrained(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            else:
                peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type_name].from_pretrained(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # load weights if any
        path = (
            os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else model_id
        )

        if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
            use_safetensors = True
        elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
            use_safetensors = False
        else:
            has_remote_safetensors_file = hub_file_exists(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                revision=hf_hub_download_kwargs.get("revision", None),
                repo_type=hf_hub_download_kwargs.get("repo_type", None),
            )
            use_safetensors = has_remote_safetensors_file

            if has_remote_safetensors_file:
                # Priority 1: load safetensors weights
                filename = hf_hub_download(
                    model_id,
                    SAFETENSORS_WEIGHTS_NAME,
                    **hf_hub_download_kwargs,
                )
            else:
                try:
                    filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
                except EntryNotFoundError:
                    raise ValueError(
                        f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                        f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                    )

        if use_safetensors:
            adapters_weights = safe_load_file(filename, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # load the weights into the model
        if self.peft_config[adapter_name].peft_type in (DeltaTunerType.SSF):
            load_result = set_deltatuner_model_state_dict(self, adapters_weights, adapter_name=adapter_name)    
        else:
            load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            if isinstance(self.peft_config[adapter_name], PromptLearningConfig):
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

class DelatunerModelForCausalLM(DeltaTunerModel):
    def __init__(self, model: PeftModel, peft_config: PeftConfig, adapter_name: str = "default", denas_config: DeltaTunerArguments = None, tokenizer: AutoTokenizer = None):
        super().__init__(model, peft_config, adapter_name, denas_config, tokenizer)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if self.base_model.config.model_type == "mpt":
            if inputs_embeds is not None:
                raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        
        return model_kwargs
        