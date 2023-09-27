import os, sys, time, math
import torch
from torch import nn
import numpy as np
import gc
import torch
from torchsummary import summary
from torch.nn.modules import linear
from transformers import pytorch_utils

from .utils import ATTN_LAYER_STRUCTURE_NAME, LINEAR_LAYER_STRUCTURE_NAME, LINEAR_MODULE_TYPE, LINEAR_LORA_MODULE_TYPE

# TODO: currently specific support for one pure lora layer
def network_weight_gaussian_init(model):
    with torch.no_grad():
        for k, m in model.named_modules():
            if any(isinstance(m, module_type) for module_type in LINEAR_LORA_MODULE_TYPE):
                if hasattr(m, "lora_A") and "default" in m.lora_A.keys():
                    nn.init.kaiming_uniform_(m.lora_A["default"].weight, a=math.sqrt(5), mode='fan_in')
                    if hasattr(m.lora_A["default"], "bias") and m.lora_A["default"].bias is not None:
                        nn.init.zeros_(m.lora_A["default"].bias)
                if hasattr(m, "lora_B") and "default" in m.lora_B.keys():
                    nn.init.kaiming_uniform_(m.lora_B["default"].weight, a=math.sqrt(5), mode='fan_out')
                    if hasattr(m.lora_B["default"], "bias") and m.lora_B["default"].bias is not None:
                        nn.init.zeros_(m.lora_B["default"].bias)
    return model

# TODO: separate domain specific ops
def get_linear_layer_metric_array(net, metric):
    metric_array = []
    for k, module in net.named_modules():
        if not any(name in k for name in LINEAR_LAYER_STRUCTURE_NAME):
            continue
        else:
            target = net.get_submodule(k)
            if any(isinstance(target, module_type) for module_type in LINEAR_MODULE_TYPE):
                metric_array.append(metric(target))
    return metric_array

def get_attn_layer_metric_array(net, metric):
    metric_array = []
    for k, module in net.named_modules():
        if not any(name in k for name in ATTN_LAYER_STRUCTURE_NAME):
            continue
        else:
            target = net.get_submodule(k)
            if any(isinstance(target, module_type) for module_type in LINEAR_MODULE_TYPE):
                metric_array.append(metric(target))
    return metric_array

def compute_diversity_score(net):
    
    # select the gradients that we want to use for search/prune
    def nuc_norm(layer):
        if layer.weight.grad is not None:
            return torch.norm(layer.weight, p="nuc") * torch.norm(layer.weight.grad, p="nuc")
        else:
            return torch.zeros_like(layer.weight)

    diversity_score_list = get_attn_layer_metric_array(net, nuc_norm)
    return diversity_score_list

def compute_saliency_score(net):
    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            #param.abs_()
        return signs
    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    #signs = linearize(net)
    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_linear_layer_metric_array(net, synflow)
    grads_abs.extend(get_attn_layer_metric_array(net, synflow))
    # apply signs of all params
    #nonlinearize(net, signs)
    return grads_abs
