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