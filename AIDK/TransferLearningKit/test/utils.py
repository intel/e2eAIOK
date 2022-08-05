#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/5/2022 9:13 AM
import torch

def tensor_near_equal(tensor1,tensor2,threshold = 1e-9):
    ''' if tensor1 is near equal tensor2

    :param tensor1: the first tensor
    :param tensor2: the second tensor
    :param threshold: threshold value
    :return:
    '''
    if tensor1.shape != tensor2.shape:
        return False
    return torch.max(torch.abs(tensor1 - tensor2)).item() <= threshold