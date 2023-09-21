import os, sys, time, math
import torch
from torch import nn
import numpy as np
import gc
import torch
from torchsummary import summary
from torch.nn.modules import linear
from transformers import pytorch_utils

from .utils import ATTN_LAYER_STRUCTURE_NAME, LINEAR_LAYER_STRUCTURE_NAME
from ..search.utils import network_latency, input_constructor

# TODO: separate domain specific ops
def get_linear_layer_metric_array(net, metric):
    metric_array = []
    for k, param in net.named_parameters():
        if not any(name in k for name in LINEAR_LAYER_STRUCTURE_NAME):
            continue
        else:
            if "ssf_" in k:
                metric_array.append(metric(param))
    return metric_array

def get_attn_layer_metric_array(net, metric):
    metric_array = []
    for k, param in net.named_parameters():
        if not any(name in k for name in ATTN_LAYER_STRUCTURE_NAME):
            continue
        else:
            if "ssf_" in k:
                metric_array.append(metric(param))
    return metric_array

def compute_diversity_score(net):
    
    # select the gradients that we want to use for search/prune
    def nuc_norm(param):
        if param.grad is not None:
            return torch.linalg.vector_norm(param) * torch.linalg.vector_norm(param.grad)
        else:
            return torch.zeros_like(param)

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
    def synflow(param):
        if param.grad is not None:
            return torch.abs(param * param.grad)
        else:
            return torch.zeros_like(param)

    grads_abs = get_linear_layer_metric_array(net, synflow)
    grads_abs.extend(get_attn_layer_metric_array(net, synflow))
    # apply signs of all params
    #nonlinearize(net, signs)
    return grads_abs
