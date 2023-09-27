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
