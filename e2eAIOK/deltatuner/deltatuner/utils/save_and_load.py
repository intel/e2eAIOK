# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
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

from .config import DeltaTunerType

def get_deltatuner_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the DeltaTuner model.

    Args:
        model ([`DeltaTunerModelModel`]): The DeltaTunerModel model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    
    if config.peft_type == DeltaTunerType.SSF:
        bias = config.bias
        if bias is None or bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "ssf_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "ssf_" in k or "bias" in k}
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_deltatuner_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the DeltaTuner model.

    Args:
        model ([`DeltaTunerModel`]): The DeltaTuner model.
        peft_model_state_dict (`dict`): The state dict of the DeltaTuner model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (DeltaTunerType.SSF):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            peft_model_state_dict[k] = v
    else:
        raise NotImplementedError

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    return load_result
