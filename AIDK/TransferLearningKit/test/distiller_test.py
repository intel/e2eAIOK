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
from engine_core.distiller import BasicDistiller, KD, DKD
import random

torch.manual_seed(0)
random.seed(32)

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

class TestKD:
    ''' Test KD
    
    '''
    def _get_kwargs(self):
        kwargs = {
            "pretrained_model": torchvision.models.resnet18(pretrained=True),
            "temperature": 4.0,
            "is_frozen": True,
        }
        return kwargs
    def test_create(self):
        ''' test create a distiller

        :return:
        '''
        kwargs = self._get_kwargs()
        ################ frozen create ###################
        distiller = KD(**kwargs)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == False

        ################ unfrozen create ###################
        kwargs["pretrained_model"] = torchvision.models.resnet18(pretrained=True)
        kwargs["is_frozen"] = False
        distiller = KD(**kwargs)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == True

    def test_forward(self):
        ''' test forward

        :return:
        '''
        kwargs = self._get_kwargs()
        num_classes = 1000
        bath_size = 16
        distiller = KD(**kwargs)
        x = torch.zeros([bath_size,3,224,224])
        y = distiller(x)
        assert y.shape == torch.Size([bath_size,num_classes])

    def test_loss(self):
        ''' test loss

        :return:
        '''
        kwargs = self._get_kwargs()
        distiller = KD(**kwargs)
        num_classes = 1000
        bath_size = 16
        logits1 = torch.ones([bath_size,num_classes])
        logits2 = torch.zeros([bath_size, num_classes])
        logits3 = torch.randn([bath_size,num_classes])
        logits4 = 1 - logits3
        assert torch.abs(distiller.loss(logits3, logits3)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits1, logits2)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits3, logits4)).item() >= 1


class TestDKD:
    ''' Test DKD
    
    '''
    def _get_kwargs(self):
        kwargs = {
            "pretrained_model": torchvision.models.resnet18(pretrained=True),
            "alpha": 1.0,
            "beta": 8.0,
            "temperature": 4.0,
            "warmup": 20,
            "is_frozen": True,
        }
        return kwargs

    def test_create(self):
        ''' test create a distiller

        :return:
        '''
        kwargs = self._get_kwargs()
        ################ frozen create ###################
        distiller = DKD(**kwargs)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == False

        ################ unfrozen create ###################
        kwargs["pretrained_model"] = torchvision.models.resnet18(pretrained=True)
        kwargs["is_frozen"] = False
        distiller = DKD(**kwargs)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == True

    def test_forward(self):
        ''' test forward

        :return:
        '''
        kwargs = self._get_kwargs()
        num_classes = 1000
        bath_size = 16
        distiller = DKD(**kwargs)
        x = torch.zeros([bath_size,3,224,224])
        y = distiller(x)
        assert y.shape == torch.Size([bath_size,num_classes])

    def test_loss(self):
        ''' test loss

        :return:
        '''
        kwargs = self._get_kwargs()
        distiller = DKD(**kwargs)
        num_classes = 1000
        bath_size = 16
        logits1 = torch.ones([bath_size,num_classes])
        logits2 = torch.zeros([bath_size, num_classes])
        logits3 = torch.randn([bath_size,num_classes])
        target = torch.Tensor([random.randint(0,1000) for i in range(16)]).long()
        assert torch.abs(distiller.loss(logits3, logits3, epoch=20,target=target)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits1, logits2, epoch=20,target=target)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits1, logits3, epoch=20,target=target)).item() >= 1