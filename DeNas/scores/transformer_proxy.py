import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import gc
from scores.basic_utils import *
import torch
from module.Linear_super import LinearSuper
from module.layernorm_super import LayerNormSuper
from module.multihead_super import AttentionSuper
from cv.supernet_transformer import TransformerEncoderLayer

from torchsummary import summary

def get_linear_layer_metric_array(net, metric):
    metric_array = []
    for layer in net.modules():
        if isinstance(layer, LinearSuper):
            metric_array.append(metric(layer))
        elif isinstance(layer, TransformerEncoderLayer):
            metric_array.append(metric(layer.fc1))
            metric_array.append(metric(layer.fc2))        
    return metric_array

def get_attn_layer_metric_array(net, metric):
    score = 0 

    metric_array = []
    for layer in net.modules():
        if isinstance(layer, TransformerEncoderLayer):
            metric_array.append(metric(layer.attn.qkv))
            metric_array.append(metric(layer.attn.proj))
    return metric_array


def compute_diversity_score(net, inputs):


    # Compute gradients with input of 1s
    net.zero_grad()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim)

    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def nuc_norm(layer):
        if layer.weight.grad is not None:
            return torch.norm(layer.weight) * torch.norm(layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    disversity_score_list = get_attn_layer_metric_array(net, nuc_norm)


    
    return disversity_score_list



def compute_sliency_score(net, inputs):
    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim)

    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_linear_layer_metric_array(net, synflow)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs



def do_compute_nas_score_transformer(model_type, model, resolution, batch_size, mixup_gamma):
    dtype = torch.float32
    network_weight_gaussian_init(model)
    model.train()
    model.requires_grad_(True)

    model.zero_grad()
    input = torch.randn(size=[batch_size, 3, resolution, resolution],  dtype=dtype)
    disversity_score_list = compute_diversity_score(model, input)
    print(F"len:{len(disversity_score_list)}")

    disversity_score = 0
    for grad_abs in disversity_score_list:
        if len(grad_abs.shape) == 0:
            disversity_score += grad_abs
        elif len(grad_abs.shape) == 2:
            disversity_score += float(torch.mean(torch.sum(grad_abs, dim=[1])))


    grads_abs_list = compute_sliency_score(net=model, inputs=input)
   
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('only support grad shape of 4 or 2')
    
    

    nas_score = disversity_score + score


    return nas_score


