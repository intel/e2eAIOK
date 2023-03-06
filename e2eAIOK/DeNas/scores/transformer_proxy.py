import os, sys, time, math
import torch
from torch import nn
import numpy as np
import gc
import torch
from e2eAIOK.DeNas.utils import NETWORK_LATENCY
from torch.nn.modules import linear
from transformers import pytorch_utils

# TODO: separate domain specific ops
from e2eAIOK.DeNas.module.cv.Linear_super import LinearSuper
from e2eAIOK.DeNas.module.cv.layernorm_super import LayerNormSuper
from e2eAIOK.DeNas.module.cv.multihead_super import AttentionSuper
from e2eAIOK.DeNas.module.cv.embedding_super import PatchembedSuper
from e2eAIOK.DeNas.cv.supernet_transformer import TransformerEncoderLayer
from e2eAIOK.DeNas.nlp.supernet_bert import SuperBertEncoder
from e2eAIOK.DeNas.module.asr.encoder import TransformerEncoder
from e2eAIOK.DeNas.module.asr.attention import MultiheadAttention
from e2eAIOK.DeNas.module.asr.linear import Linear
from e2eAIOK.DeNas.thirdparty.supernet_hf import SuperHFModel
from e2eAIOK.DeNas.thirdparty.utils import input_construtor, LINEAR_LAYER_STRUCTURE, ATTN_LAYER_STRUCTURE

from torchsummary import summary

def network_weight_gaussian_init(net, model_type):
    with torch.no_grad():
        if model_type == "transformer":   
            for m in net.modules():
                if isinstance(m, LinearSuper):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, TransformerEncoderLayer):
                    nn.init.ones_(m.fc1.weight)
                    nn.init.zeros_(m.fc1.bias)
                    nn.init.ones_(m.fc2.weight)
                    nn.init.zeros_(m.fc2.bias)
                    nn.init.ones_(m.attn_layer_norm.weight)
                    nn.init.zeros_(m.attn_layer_norm.bias)
                    nn.init.ones_(m.ffn_layer_norm.weight)
                    nn.init.zeros_(m.ffn_layer_norm.bias)
                elif isinstance(m, LayerNormSuper):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, PatchembedSuper):
                    nn.init.normal_(m.proj.weight)
                    if hasattr(m.proj, 'bias') and m.proj.bias is not None:
                        nn.init.zeros_(m.proj.bias)
                else:
                    continue
    return net

def get_linear_layer_metric_array(model_type, net, metric):
    metric_array = []
    for layer in net.modules():
        if model_type == "transformer":
            if isinstance(layer, LinearSuper):
                metric_array.append(metric(layer))
            elif isinstance(layer, TransformerEncoderLayer):
                metric_array.append(metric(layer.fc1))
                metric_array.append(metric(layer.fc2))
        elif model_type == "bert":
            if isinstance(layer, SuperBertEncoder):
                for sub_layer in layer.layers:
                    metric_array.append(metric(sub_layer.intermediate.dense))
                    metric_array.append(metric(sub_layer.output.dense))
        elif model_type == "asr":
            if isinstance(layer, TransformerEncoder):
                for sub_layer in layer.layers:
                    metric_array.append(metric(sub_layer.pos_ffn.fc1))
                    metric_array.append(metric(sub_layer.pos_ffn.fc2))
        elif model_type == "hf":
            for k in LINEAR_LAYER_STRUCTURE:
                if k in type(layer).__name__.lower():
                    for sub_layer in layer.modules():
                        if isinstance(sub_layer, LINEAR_LAYER_STRUCTURE[k]):
                            metric_array.append(metric(sub_layer))
    return metric_array

def get_attn_layer_metric_array(model_type, net, metric):
    def func(weight):
        if weight.grad is not None:
            return torch.norm(weight) * torch.norm(weight.grad)
        else:
            return torch.zeros_like(weight)
    score = 0 
    metric_array = []
    for layer in net.modules():
        if model_type == "transformer":
            if isinstance(layer, TransformerEncoderLayer):
                metric_array.append(metric(layer.attn.qkv))
                metric_array.append(metric(layer.attn.proj))
        elif model_type == "bert":
            if isinstance(layer, SuperBertEncoder):
                for sub_layer in layer.layers:
                    metric_array.append(metric(sub_layer.attention.self.query))
                    metric_array.append(metric(sub_layer.attention.self.key))
                    metric_array.append(metric(sub_layer.attention.self.value))
                    metric_array.append(metric(sub_layer.output.dense))
        elif model_type == 'asr':
            if isinstance(layer, TransformerEncoder):
                for sub_layer in layer.layers:
                    metric_array.append(func(sub_layer.self_att.att.in_proj_weight))
                    metric_array.append(metric(sub_layer.self_att.att.out_proj))
        elif model_type == 'hf':
            for k in ATTN_LAYER_STRUCTURE:
                if k in type(layer).__name__.lower():
                    for sub_layer in layer.modules():
                        if isinstance(sub_layer, ATTN_LAYER_STRUCTURE[k]):
                            metric_array.append(metric(sub_layer))
    return metric_array


def compute_diversity_score(model_type, net, *inputs):

    # Compute gradients with input of 1s
  
    net.zero_grad()
    if model_type == "transformer":
        input_dim = list(inputs[0][0, :].shape)
        inputs = torch.ones([1] + input_dim)
        output = net.forward(inputs)
    elif model_type == "bert":
        input_ids, input_masks, input_segments = inputs[0]['input_ids'], inputs[0]['attention_mask'], inputs[0]['token_type_ids']
        output, pooled_output = net.forward(input_ids, input_masks, input_segments)
    elif model_type == "asr":
        output, _ = net.encode(inputs[0])
    elif model_type == "hf":
        output = net(**inputs[0])
        output = output.last_hidden_state
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def nuc_norm(layer):
        if layer.weight.grad is not None:
            return torch.norm(layer.weight) * torch.norm(layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    disversity_score_list = get_attn_layer_metric_array(model_type, net, nuc_norm)
    
    return disversity_score_list


def compute_saliency_score(model_type, net, *inputs):
    
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
    net.zero_grad()

    if model_type == "transformer":
    # Compute gradients with input of 1s
        input_dim = list(inputs[0][0, :].shape)
        inputs = torch.ones([1] + input_dim)
        output = net.forward(inputs)
    elif model_type == "bert":
        input_ids, input_masks, input_segments = inputs[0]['input_ids'], inputs[0]['attention_mask'], inputs[0]['token_type_ids']
        output, pooled_output = net.forward(input_ids, input_masks, input_segments)
    elif model_type == "asr":
        output, _ = net.encode(inputs[0])
    elif model_type == "hf":
        output = net(**inputs[0])
        output = output.last_hidden_state

    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_linear_layer_metric_array(model_type, net, synflow)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs

def do_compute_nas_score_transformer(model_type, model, resolution, batch_size, mixup_gamma, expressivity_weight=0, complexity_weight=0, diversity_weight=0, saliency_weight=0, latency_weight=0):
    
    expressivity_score = 0
    complexity_score = 0
    network_weight_gaussian_init(model,model_type)
    model.train()
    model.requires_grad_(True)
    model.zero_grad()
    

    if model_type == "transformer":
        dtype = torch.float32
        input = torch.randn(size=[batch_size, 3, resolution, resolution],  dtype=dtype)
        disversity_score_list = compute_diversity_score(model_type, model, input)
    elif model_type == "bert":
        input = input_construtor(batch_size, resolution)
        disversity_score_list = compute_diversity_score(model_type, model, input)
    elif model_type == "asr":
        input = torch.randn(size=[batch_size, 400, 20, 64])
        disversity_score_list = compute_diversity_score(model_type, model, input)
    elif model_type == "hf":
        input = input_construtor(batch_size, resolution)
        disversity_score_list = compute_diversity_score(model_type, model, input)
    disversity_score = 0
    for grad_abs in disversity_score_list:
        if len(grad_abs.shape) == 0:
            disversity_score += float(grad_abs)
        elif len(grad_abs.shape) == 2:
            disversity_score += float(torch.mean(torch.sum(grad_abs, dim=[1])))

    if model_type == "transformer":
        grads_abs_list = compute_saliency_score(model_type, model, input)
    elif model_type == "bert":
        input = input_construtor(batch_size, resolution)
        grads_abs_list = compute_saliency_score(model_type, model, input)
    elif model_type == "asr":
        grads_abs_list = compute_saliency_score(model_type, model, input)
    elif model_type == "hf":
        input = input_construtor(batch_size, resolution)
        grads_abs_list = compute_saliency_score(model_type, model, input)
   
    saliency_score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            saliency_score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            saliency_score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('only support grad shape of 4 or 2')
    if model_type == "transformer":
        latency = NETWORK_LATENCY[model_type](model=model, batch_size=batch_size,
                                                        resolution=resolution,
                                                        in_channels=3, gpu=None, repeat_times=3,
                                                        fp16=False)    
    elif model_type == "bert":
        latency = NETWORK_LATENCY[model_type](model=model, batch_size=batch_size, max_seq_length=resolution, gpu=None, infer_cnt=10.)
    elif model_type == "hf":
        latency = NETWORK_LATENCY[model_type](model=model, batch_size=batch_size, max_seq_length=resolution, gpu=None, infer_cnt=10.)
    else:
        latency = 0
    score = (expressivity_score*expressivity_weight 
                    + complexity_score*complexity_weight 
                    + disversity_score*diversity_weight 
                    + saliency_score*saliency_weight)
    nas_score = score/(1 + latency*latency_weight)
    return nas_score, score, latency
