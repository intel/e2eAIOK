import enum
from dataclasses import  field
from typing import List, Optional, Tuple, Union
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

class DeltaTunerType(str, enum.Enum):
    SSF = "SSF"

TRANSFORMERS_MODELS_TO_SSF_TARGET_MODULES_MAPPING = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
TRANSFORMERS_MODELS_TO_SSF_TARGET_MODULES_MAPPING.update({
    "llama": ["q_proj", "v_proj"],
    "mpt": ["Wqkv","out_proj","up_proj","down_proj"]
})

BEST_MODEL_STRUCTURE_DEFAULT_NAME = "best_model_structure.txt"