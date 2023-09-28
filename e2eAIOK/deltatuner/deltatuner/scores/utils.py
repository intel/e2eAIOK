import torch
from transformers import pytorch_utils
from peft.tuners.lora import Linear, Conv1D, LoraLayer

LINEAR_LAYER_STRUCTURE_NAME = ["output", "intermediate", "mlp", "ffn"]

ATTN_LAYER_STRUCTURE_NAME = ["attn", "atten"]

LINEAR_MODULE_TYPE =  [torch.nn.Linear, pytorch_utils.Conv1D, Linear, Conv1D]

LINEAR_LORA_MODULE_TYPE = [LoraLayer]