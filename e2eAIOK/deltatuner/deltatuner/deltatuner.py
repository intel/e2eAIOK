import logging
from peft import LoraConfig, PeftModel, LoraModel
from .tuner import SSFConfig

from .deltatuner_model import DeltaTunerModel
from .deltatuner_args import DeltaTunerArguments
from .mapping import MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING

logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deltatuner')

SUPPORTED_ALGO = ["auto", "lora", "ssf"]

def optimize(model, tokenizer=None, adapter_name: str="default", deltatuning_args: DeltaTunerArguments=None, **kwargs) -> DeltaTunerModel:
    if deltatuning_args is None:
        deltatuning_args = DeltaTunerArguments(**kwargs)

    algo = deltatuning_args.algo
    if algo not in SUPPORTED_ALGO:
        raise NotImplementedError("Current algorithm {} is not supported in deltatuner. ".format(algo))
    
    if algo == "auto":
        if "mpt" in model.config.model_type.lower() or not isinstance(model, PeftModel):
            algo = "ssf"
        else:
            algo = "lora"

    if algo == "ssf" and not isinstance(model, PeftModel):
        peft_config = SSFConfig(target_modules=deltatuning_args.ssf_target_module, bias=None, task_type=deltatuning_args.task_type)
        peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    elif algo == "lora" and isinstance(model, PeftModel) and isinstance(model.peft_config[adapter_name], LoraConfig):
        peft_config = model.peft_config[adapter_name]
        if not deltatuning_args.denas:
            return model
    else:
        raise NotImplementedError("Current algorithm {} does not support {} type of model in deltatuner. Please specify the right algo with the type of model.".format(algo, type(model)))

    whether_denas = "enable" if deltatuning_args.denas else "disable"
    logging.info("The user specifies or automatically adjasts {} algo on model {} with {} denas".format(algo, model.config.model_type, whether_denas))

    if isinstance(peft_config, LoraConfig) or isinstance(peft_config, SSFConfig):
        if peft_config.task_type not in MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING:
            model = DeltaTunerModel(model, peft_config, adapter_name, deltatuning_args, tokenizer)
        else:
            model = MODEL_TYPE_TO_DELTATUNER_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name, deltatuning_args, tokenizer)
        return model
    else:
        raise NotImplementedError("Current algorithm {} is not supported in deltatuner. ".format(algo))