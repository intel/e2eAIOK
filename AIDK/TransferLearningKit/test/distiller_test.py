#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/4/2022 1:35 PM

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"test"))
import torch
import torchvision
from engine_core.distiller.basic_distiller import BasicDistiller

class TestBasicDistiller:
    ''' Test BasicDistiller
    
    '''
    def test_create(self):
        ''' test create a distiller

        :return:
        '''
        ################ frozen create ###################
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), True)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == False

        ################ unfrozen create ###################
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), False)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == True

    def test_forward(self):
        ''' test forward

        :return:
        '''
        num_classes = 1000
        bath_size = 16
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), True)
        x = torch.zeros([bath_size,3,224,224])
        y = distiller(x)
        assert y.shape == torch.Size([bath_size,num_classes])

    def test_loss(self):
        ''' test loss

        :return:
        '''
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), True)
        num_classes = 1000
        bath_size = 16
        logits1 = torch.ones([bath_size,num_classes])
        logits2 = torch.zeros([bath_size, num_classes])
        logits3 = torch.randn([bath_size,num_classes])
        assert torch.abs(distiller.loss(logits3, logits3)).item() <= 1e-9
        assert torch.abs(distiller.loss(logits1, logits2) - 1.0).item() <= 1e-9