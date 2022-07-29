#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/27/2022 3:20 PM
import logging
import torch

def accuracy(output,label):
    pred = output.data.cpu().max(1)[1]
    label = label.data.cpu()

    if pred.shape != label.shape:
        logging.error('pred shape[%s] and label shape[%s] not match' % (pred.shape, label.shape))
        raise RuntimeError('pred shape[%s] and label shape[%s] not match' % (pred.shape, label.shape))
    return torch.mean((pred == label).float())

MetricMap = {
    'acc' : accuracy,
    'accuracy' : accuracy,
}
