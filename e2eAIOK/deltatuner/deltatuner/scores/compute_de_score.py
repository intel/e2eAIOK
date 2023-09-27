

import os, sys, time, logging
import torch
from torch import nn
import numpy as np
import gc

import torch

from ..utils import DeltaTunerType
from ..search.utils import network_latency, input_constructor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DENAS')

def do_compute_nas_score_transformer(model, tokenizer, resolution, batch_size, expressivity_weight=0, complexity_weight=0, diversity_weight=0, saliency_weight=0, latency_weight=0,peft_type=None):
    if peft_type in (DeltaTunerType.SSF):
        from .transformer_proxy_ssf import compute_diversity_score, compute_saliency_score
    else:
        from .transformer_proxy import network_weight_gaussian_init, compute_diversity_score, compute_saliency_score
        network_weight_gaussian_init(model)
    expressivity_score = 0
    complexity_score = 0
    model.train()
    #model.requires_grad_(True)
    model.zero_grad() 
    
    input, _ = input_constructor(batch_size=batch_size, resolution=resolution, tokenizer=tokenizer)
    output = model(**input)
    output = output.logits
    torch.sum(output).backward()

    diversity_score_list = compute_diversity_score(model)
    diversity_score = 0
    for grad_abs in diversity_score_list:
        if len(grad_abs.shape) == 0:
            diversity_score += float(grad_abs)
        elif len(grad_abs.shape) == 2:
            diversity_score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError("diversity_score only support grad shapre of 0 or 2")

    grads_abs_list = compute_saliency_score(model)
    saliency_score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            saliency_score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            saliency_score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        elif len(grad_abs.shape) == 1:
            saliency_score += float(torch.mean(grad_abs))
        else:
            raise RuntimeError('saliency_score only support grad shape of 1, 2 or 4')
    
    latency = 0
    if latency_weight != 0:
        latency = network_latency(model=model, tokenizer=tokenizer, batch_size=batch_size)
    saliency_score_scale = 10.**5 if peft_type in (DeltaTunerType.SSF) else 10.**9
    score = (expressivity_score*expressivity_weight / 10.**9
                    + complexity_score*complexity_weight / 10.**9
                    + diversity_score*diversity_weight / 10.**9
                    + saliency_score*saliency_weight / saliency_score_scale)
    logger.info("saliency score:{}, diversity score:{}, latency_score:{}".format(saliency_score/saliency_score_scale, diversity_score/10.**9, latency))
    nas_score = score/(1 + latency*latency_weight)
    return nas_score, score, latency

def do_compute_nas_score(model, tokenizer, resolution, batch_size, mixup_gamma, expressivity_weight=0, complexity_weight=0, diversity_weight=0, saliency_weight=0, latency_weight=0,peft_type=None):
    nas_score, score, latency = do_compute_nas_score_transformer(model, tokenizer, resolution, batch_size, expressivity_weight, complexity_weight, diversity_weight, saliency_weight, latency_weight,peft_type)
    return nas_score, score, latency