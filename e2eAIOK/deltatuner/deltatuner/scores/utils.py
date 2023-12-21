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

import torch
from transformers import pytorch_utils
from peft.tuners.lora import Linear, Conv1D, LoraLayer

LINEAR_LAYER_STRUCTURE_NAME = ["output", "intermediate", "mlp", "ffn"]

ATTN_LAYER_STRUCTURE_NAME = ["attn", "atten"]

LINEAR_MODULE_TYPE =  [torch.nn.Linear, pytorch_utils.Conv1D, Linear, Conv1D]

LINEAR_LORA_MODULE_TYPE = [LoraLayer]