

import os, sys, time
import torch
from torch import nn
import numpy as np
import gc

import torch

from e2eAIOK.DeNas.scores.basic_utils import network_weight_gaussian_init, get_ntk_n, compute_synflow_per_weight
from e2eAIOK.DeNas.scores.transformer_proxy import do_compute_nas_score_transformer

def do_compute_nas_score_cnn(model_type, model, resolution, batch_size, mixup_gamma, expressivity_weight=0, complexity_weight=0, diversity_weight=0, saliency_weight=0, latency_weight=0):
    disversity_score = 0
    latency = 0
    torch.manual_seed(12345)
    dtype = torch.float32
    network_weight_gaussian_init(model)
    with torch.no_grad():
        input = torch.randn(size=[batch_size, 3, resolution, resolution],  dtype=dtype)
        input2 = torch.randn(size=[batch_size, 3, resolution, resolution], dtype=dtype)
        mixup_input = input + mixup_gamma * input2

        output = model.forward_pre_GAP(input)
        mixup_output = model.forward_pre_GAP(mixup_input)

        expressivity_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
        expressivity_score = torch.log(torch.mean(expressivity_score))

    complexity_score = get_ntk_n([model], recalbn=0, train_mode=True, num_batch=1,
                           batch_size=batch_size, image_size=resolution)[0]

    model.train()
    model.requires_grad_(True)

    model.zero_grad()


    grads_abs_list = compute_synflow_per_weight(model_type, net=model, inputs=input, mode='')
   
    saliency_score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            saliency_score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            saliency_score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('only support grad shape of 4 or 2')
    
    score = expressivity_score*expressivity_weight/(saliency_score*saliency_weight) \
                    - complexity_score*complexity_weight \
                    + disversity_score*diversity_weight
    nas_score = score/(1 + latency*latency_weight)
    return nas_score, score, latency


def do_compute_nas_score(model_type, model, resolution, batch_size, mixup_gamma, expressivity_weight=0, complexity_weight=0, diversity_weight=0, saliency_weight=0, latency_weight=0):
    if model_type == "cnn":
        nas_score, score, latency = do_compute_nas_score_cnn(model_type, model, resolution, batch_size, mixup_gamma, expressivity_weight, complexity_weight, diversity_weight, saliency_weight, latency_weight)
        return nas_score, score, latency
    else:
        nas_score, score, latency = do_compute_nas_score_transformer(model_type, model, resolution, batch_size, mixup_gamma, expressivity_weight, complexity_weight, diversity_weight, saliency_weight, latency_weight)
        return nas_score, score, latency

