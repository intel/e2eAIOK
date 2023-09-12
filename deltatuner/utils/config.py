import enum
from dataclasses import  field
from typing import List, Optional, Tuple, Union

class DeltaTunerType(str, enum.Enum):
    SSF = "SSF"

TRANSFORMERS_MODELS_TO_SSF_TARGET_MODULES_MAPPING = {
    "llama": ["q_proj", "v_proj"],
    "mpt": ["Wqkv"]
}