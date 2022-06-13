

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import gc

import torch

from scores.basic_utils import *
from scores.transformer_proxy import do_compute_nas_score_transformer


def do_compute_nas_score_cnn(model_type, model, resolution, batch_size, mixup_gamma):

    dtype = torch.float32
    network_weight_gaussian_init(model)
    with torch.no_grad():
        input = torch.randn(size=[batch_size, 3, resolution, resolution],  dtype=dtype)
        input2 = torch.randn(size=[batch_size, 3, resolution, resolution], dtype=dtype)
        mixup_input = input + mixup_gamma * input2

        output = model.forward_pre_GAP(input)
        mixup_output = model.forward_pre_GAP(mixup_input)

        nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
        nas_score = torch.mean(nas_score)
    

    model.train()
    model.requires_grad_(True)

    model.zero_grad()


    grads_abs_list = compute_synflow_per_weight(model_type, net=model, inputs=input, mode='')
   
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('only support grad shape of 4 or 2')
    
    

    nas_score = torch.log(nas_score)/( 1 * score)



    return nas_score



def do_compute_nas_score_cnn_plus(model_type, model, resolution, batch_size, mixup_gamma):
    dtype = torch.float32
    network_weight_gaussian_init(model)
    with torch.no_grad():
        input = torch.randn(size=[batch_size, 3, resolution, resolution],  dtype=dtype)
        input2 = torch.randn(size=[batch_size, 3, resolution, resolution], dtype=dtype)
        mixup_input = input + mixup_gamma * input2

        output = model.forward_pre_GAP(input)
        mixup_output = model.forward_pre_GAP(mixup_input)

        nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
        nas_score = torch.mean(nas_score)
    ntk_score = get_ntk_n([model], recalbn=0, train_mode=True, num_batch=1,
                           batch_size=batch_size, image_size=resolution)[0]

    model.train()
    model.requires_grad_(True)

    model.zero_grad()


    grads_abs_list = compute_synflow_per_weight(model_type, net=model, inputs=input, mode='')
   
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('only support grad shape of 4 or 2')
    
    

    nas_score = torch.log(nas_score) / (1 * score) + (-1 * ntk_score)


    return nas_score


def do_compute_nas_score(model_type, model, resolution, batch_size, mixup_gamma):
    if model_type == "cnn":
        return do_compute_nas_score_cnn(model_type, model, resolution, batch_size, mixup_gamma)
    elif model_type == "transformer":
        return do_compute_nas_score_transformer(model_type, model, resolution, batch_size, mixup_gamma)

