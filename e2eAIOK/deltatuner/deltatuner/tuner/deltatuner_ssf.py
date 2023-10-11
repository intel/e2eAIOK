import os
import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

# from peft.import_utils import is_bnb_available, is_bnb_4bit_available
from peft.utils import (
    CLAMP_QUANTILE,
    COMMON_LAYERS_PATTERN,
    ModulesToSaveWrapper,
    PeftConfig,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

from ..utils import DeltaTunerType, TRANSFORMERS_MODELS_TO_SSF_TARGET_MODULES_MAPPING

# if is_bnb_available():
#     import bitsandbytes as bnb

@dataclass
class SSFConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SSFModel`].

    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply ssf to.
        bias (`str`): Bias type for ssf. Can be 'none', 'all'. If 'all', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the ssf transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the ssf
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        modules_to_save (`List[str]`):List of modules apart from SSF layers to be set as trainable
            and saved in the final checkpoint.
    """

    bias: str = field(default="none", metadata={"help": "Bias type for SSF. Can be 'none', 'all'"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ssf."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from SSF layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = DeltaTunerType.SSF

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        from peft.utils import CONFIG_NAME
        
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        config_cls = cls

        config = config_cls(**class_kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


class DeltaSSFSearchSpace:
    @classmethod
    def generate_search_space(cls, model, denas_config):
        search_space = {}
        search_space_name = ["num_hidden_layers"]
        for i in range(getattr(model.config, denas_config.layer_name)):
            search_space["num_hidden_layers_{}".format(i)] = [0,1]
        return search_space, search_space_name
    
    @classmethod
    def generate_supernet(cls, model, denas_config, search_space):
        supernet_config = {}
        supernet_config["num_hidden_layers"] = [search_space["num_hidden_layers_{}".format(i)][-1] for i in range(getattr(model.config, denas_config.layer_name))]
        return supernet_config

class DeltaSSFModel(torch.nn.Module):
    """
    Creates SSF model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SSFConfig`]): The configuration of the SSF model.

    Returns:
        `torch.nn.Module`: The SSF model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SSFConfig`]): The configuration of the SSF model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.adapter_name = adapter_name
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = getattr(self.model, "config", {"model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()

            config = self._prepare_ssf_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        mark_only_ssf_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            # _freeze_adapter(self.model, adapter_name)
            for n, p in self.model.named_parameters():
                p.requires_grad = False

    # def _check_quantization_dependency(self):
    #     loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
    #     loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
    #     if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
    #         raise ImportError(
    #             "To use SSF with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
    #             "You can install it with `pip install bitsandbytes`."
    #         )

    def _check_target_module_exists(self, ssf_config, key):
        if isinstance(ssf_config.target_modules, str):
            target_module_found = re.fullmatch(ssf_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in ssf_config.target_modules)
            is_using_layer_indexes = getattr(ssf_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(ssf_config, "layers_pattern", None)
            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(ssf_config.layers_to_transform, int):
                            target_module_found = layer_index == ssf_config.layers_to_transform
                        else:
                            target_module_found = layer_index in ssf_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_new_module(self, ssf_config, adapter_name, target):
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` are supported."
            )
        new_module = Linear(target, out_features)

        return new_module

    def _find_and_replace(self, adapter_name):
        ssf_config = self.peft_config[adapter_name]
        # self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(ssf_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            new_module = self._create_new_module(ssf_config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {ssf_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        
        # dispatch to correct device
        if hasattr(old_module, "weight"):
            for name, module in new_module.named_modules():
                module.to(old_module.weight.device)

    def _create_org_module(self, target):
        if isinstance(target, Linear):
            new_module = target.get_layer(merge=False)
        elif isinstance(target, torch.nn.Linear):
            new_module = target
        else:
            raise ValueError(
                f"Target module {target} is not supported. ")
        return new_module

    def set_sample_config(self, model_config):
        ssf_config = self.peft_config[self.adapter_name]
        # self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        
        for key in key_list:
            if not self._check_target_module_exists(ssf_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            l_num = [int(k) for k in key.split(".") if k.isnumeric()].pop()
            if model_config["num_hidden_layers"][l_num]:
                if not isinstance(target, SSFLayer):
                    new_module = self._create_new_module(ssf_config, self.adapter_name, target)
                    self._replace_module(parent, target_name, new_module, target)
            else:
                new_module = self._create_org_module(target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {ssf_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again.")
        mark_only_ssf_as_trainable(self.model, self.peft_config[self.adapter_name].bias)
        if self.peft_config[self.adapter_name].inference_mode:
            # _freeze_adapter(self.model, self.adapter_name)
             for n, p in self.model.named_parameters():
                p.requires_grad = False
        return self

    def merge_adapter(self):
        """
        This method merges the SSF layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, SSFLayer):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the SSF layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, SSFLayer):
                module.unmerge()

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    @staticmethod
    def _prepare_ssf_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_SSF_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_SSF_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        key_list = [key for key, _ in self.model.named_modules() if "ssf" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, SSFLayer):
                if isinstance(target, Linear):
                    new_module = target.get_layer(merge=merge)
                else:
                    raise NotImplementedError
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            # if isinstance(target, ModulesToSaveWrapper):
            #     setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def merge_and_unload(self, progressbar: bool = False):
        r"""
        This method merges the SSF layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (bool): whether to show a progressbar indicating the unload and merge process
        """
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def unload(self):
        """
        Gets back the base model by removing all the ssf modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

# had to adapt it for `ssf_only` to work
def mark_only_ssf_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "ssf_" not in n:
            p.requires_grad = False
    if bias is None or bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    else:
        raise NotImplementedError

class SSFLayer:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        
        self.ssf_scale = nn.Parameter(torch.ones(dim))
        self.ssf_shift = nn.Parameter(torch.zeros(dim))
        
        nn.init.normal_(self.ssf_scale, mean=1, std=.02)
        nn.init.normal_(self.ssf_shift, std=.02)

    def ssf_ada(self, x):
        assert self.ssf_scale.shape == self.ssf_shift.shape
        if x.shape[-1] == self.ssf_scale.shape[0]:
            return x * self.ssf_scale + self.ssf_shift
        # elif x.shape[1] == self.ssf_scale.shape[0]:
        #     return x * self.ssf_scale.view(1, -1, 1, 1) + self.ssf_shift.view(1, -1, 1, 1)
        else:
            raise ValueError('the input tensor shape does not match the shape of the scale factor.')

class Linear(nn.Module, SSFLayer):
    # SSF implemented in a dense layer
    def __init__(
        self,
        target,
        out_features: int,
    ):
        super().__init__()
        SSFLayer.__init__(self, dim = out_features)
        self.linear_sub = target
        self.merged = False
        
    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        
        #save unmerged data
        self.linear_sub_weight = self.linear_sub.weight.data.clone()
        self.linear_sub_bias = None if self.linear_sub.bias is None else self.linear_sub.bias.data.clone()

        # do merge operation
        self.linear_sub.weight.data = (self.linear_sub.weight.t() * self.ssf_scale).t()
        if self.linear_sub.bias is None:
            self.linear_sub.bias = self.ssf_shift
        else:
            self.linear_sub.bias.data = self.linear_sub.bias * self.ssf_scale + self.ssf_shift
        self.merged = True
        # print("Merge SSF layer into base model")
    
    def get_layer(self,merge=True):
        if merge and not self.merged:
            self.merge()
        if not merge and self.merged:
            self.unmerge()
        return self.linear_sub
    
    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        self.linear_sub.weight.data = self.linear_sub_weight
        if self.linear_sub_bias is None:
            self.linear_sub.bias = None
        else:
            self.linear_sub.bias.data = self.linear_sub_bias
        self.merged = False
        # print("Unmerge SSF layer from base model")

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.merged:
            result = self.linear_sub(x)
        else:
            x = self.linear_sub(x)
            result = self.ssf_ada(x)
        result = result.to(previous_dtype)
        return result
