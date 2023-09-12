from peft import get_peft_model, PeftModel, PeftConfig
from peft import LoraConfig, AdaLoraConfig
from .tuner import SSFConfig

from .deltatuner_model import DeltaTunerModel, DelatunerModelForCausalLM
from .denas_config import DeNASConfig
from .mapping import MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING

def optimize(model, adapter_name: str = "default", peft_config: PeftConfig = None, denas_config: DeNASConfig = None) -> DeltaTunerModel:
    if isinstance(model, PeftModel):
        peft_config = model.peft_config[adapter_name]
    else:
        peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    if isinstance(peft_config, LoraConfig) or isinstance(peft_config, AdaLoraConfig) or isinstance(peft_config, SSFConfig):
        if peft_config.task_type not in MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING:
            model = DeltaTunerModel(model, peft_config, adapter_name, denas_config)
        else:
            model = MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name, denas_config)
        return model
    else:
        raise NotImplementedError("Current Peft configure {} is not supported. ".format(type(peft_config)))