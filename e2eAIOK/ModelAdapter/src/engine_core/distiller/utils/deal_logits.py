#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Wang Xinyao      
# @Time   : 10/20/2022

import torch
import torch.nn
from .dataset_wrapper import DatasetWrapper
import datetime, time
import os
import numpy as np
import logging

def logits_wrap_dataset(dataset, logits_path, num_classes, save_logits, topk=0):
    dataset = DatasetWrapper(dataset,
                                logits_path=logits_path,
                                num_classes = num_classes,
                                topk=topk,
                                write=save_logits)
    return dataset

def save_check_logits(model, dataloader, epochs, start_epoch=0, topk=0, num_classes=10, model_type=None, device="cpu", save_flag=True, check_flag=False):
    if save_flag and check_flag:
        raise RuntimeError("Can not save and check logits together!")

    epoch_steps = len(dataloader)
    dataset = dataloader.dataset
    for epoch in range(start_epoch, epochs, 1):
        start_time = time.time()
        dataset.set_epoch(epoch)
        if save_flag:
            save_logits_epoch(model, dataloader, epoch_steps, topk,  model_type, device)
        if check_flag:
            check_logits_epoch(model, dataloader, epoch_steps, topk, num_classes, model_type, device)
        print(f"Epoch {epoch} took {time.time()-start_time} seconds")
        logging.info(f"Epoch {epoch} took {time.time()-start_time} seconds")

def save_logits_epoch(model, dataloader, epoch_steps, topk=0, model_type=None, device="cpu"):
    ''' save teacher logits for distiller

    :param model: the pretrained teacher model
    :param dataloader: dataloader
    :param epoch_steps: steps of one epoch
    :param topk: save top k logits, 0 means save all logits
    :param model_type: type of model
    :param device: running on cpu or gpu
    '''
    with torch.no_grad():
        model.eval()
        logits_manager = dataloader.dataset.get_manager()

        #################### iterate on dataset ##############
        for idx, ((data, label), (keys, seeds)) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            output = output[0] if isinstance(output, tuple) else output
            output = output.logits if model_type is not None and model_type.startswith("huggingface") else output
            
            if topk == 0: # save all logits
                values = output.detach().to(device='cpu', dtype=torch.float32)

                seeds = seeds.numpy()
                values = values.numpy()
                assert seeds.dtype == np.int32, seeds.dtype
                assert values.dtype == np.float32, values.dtype

                for key, seed, value in zip(keys, seeds, values):
                    bstr = seed.tobytes() + value.tobytes()
                    logits_manager.write(key, bstr)
            elif topk > 0: # only save topk logits
                softmax_prob = torch.softmax(output, -1)
                values, indices = softmax_prob.topk(k=topk, dim=-1, largest=True, sorted=True)
                values = values.detach().to(device='cpu', dtype=torch.float16)
                indices = indices.detach().to(device='cpu', dtype=torch.int16)

                seeds = seeds.numpy()
                values = values.numpy()
                indices = indices.numpy()
                assert seeds.dtype == np.int32, seeds.dtype
                assert indices.dtype == np.int16, indices.dtype
                assert values.dtype == np.float16, values.dtype

                for key, seed, indice, value in zip(keys, seeds, indices, values):
                    bstr = seed.tobytes() + value.tobytes() + indice.tobytes()
                    logits_manager.write(key, bstr)
            if idx % 10 == 0:
                dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
                print(f"{dt} save {idx}/{epoch_steps}")
                logging.info(f"{dt} save {idx}/{epoch_steps}")

def load_logits(save_values,topk=0, num_classes=10):
    if topk == 0:
        logits_value, _ = save_values
        logits_value = logits_value.float()
    elif topk > 0:
        logits_value, logits_index, _ = save_values
        logits_index = logits_index.long()
        logits_value = logits_value.float()
        minor_value = (1.0 - logits_value.sum(-1, keepdim=True)
                    ) / (num_classes - topk)
        minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
        logits_value = minor_value.scatter_(-1, logits_index, logits_value)
    return logits_value

def check_logits_epoch(model, dataloader, topk=0, num_classes=10, model_type=None, device="cpu"):
    ''' check saved teacher logits for distiller

    :param model: the evaluated model
    :param dataloader: dataloader
    :param topk: top k logits saved, 0 means save all logits
    :param num_classes: number of prediction classes
    :param model_type: type of model
    :param device: running on cpu or gpu
    '''
    with torch.no_grad():
        model.eval()  # set evaluating flag
        metric_values = {}
        #################### iterate on dataset ##############
        for idx, (data, label) in enumerate(dataloader):
            inputs, save_values = data
            inputs = inputs.to(device)
            label = label.to(device)
            batch_size = inputs.size(0)
            output = model(inputs)
            output = output[0] if isinstance(output, tuple) else output
            output = output.logits if model_type.startswith("huggingface") else output

            logits_value = output.detach().to(device='cpu', dtype=torch.float32)
            save_logits_value = load_logits(save_values, topk, num_classes)
            # softmax_prob = torch.softmax(output, -1)
            # logits_value_topk, logits_indices_topk = softmax_prob.topk(k=topk, dim=-1, largest=True, sorted=True)
            # logits_value_topk = logits_value_topk.detach().to(device='cpu', dtype=torch.float16)
            # logits_indices_topk = logits_indices_topk.detach().to(device='cpu', dtype=torch.int16)
            # metric_values["indices_diff"] = torch.count_nonzero((logits_indices_topk != save_logits_index_topk)).item() / batch_size
            # metric_values["value_topk_diff"] = (logits_value_topk - save_logits_value_topk).abs().sum().item() / batch_size
            metric_values["value_diff"] = (logits_value - save_logits_value).abs().sum().item() / batch_size
            for key in metric_values:
                print(f"{key}: {metric_values}")