"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from .tuner import SSFConfig
from typing import TYPE_CHECKING, Any, Dict
from .deltatuner_model import DeltaTunerModel, DelatunerModelForCausalLM


DELTATUNER_TYPE_TO_CONFIG_MAPPING = {
    "SSF": SSFConfig,
}

MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING = {
    "CAUSAL_LM": DelatunerModelForCausalLM
}

def get_delta_config(config_dict: Dict[str, Any]):
    """
    Returns a Delta Tuner config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return DELTATUNER_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)
