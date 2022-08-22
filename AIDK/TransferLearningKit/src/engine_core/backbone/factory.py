#!/usr/bin/python
# -*- coding: utf-8 -*-
from .lenet import LeNet
from torchvision.models import resnet18,resnet50
from .resnet_imagenet import resnet18 as resnet18_imagenet
from .resnet_imagenet import resnet50 as resnet50_imagenet
from .resnetv2 import ResNet18 as resnet18_v2
from .resnetv2 import ResNet34 as resnet34_v2
from .resnetv2 import ResNet50 as resnet50_v2
import logging
import torch

def createBackbone(backbone_name,**kwargs):
    ''' create backbone by name

    :param backbone_name: backbone name
    :param kwargs: kwargs to create backbone
    :return: a backbone model
    '''
    if backbone_name.lower() == 'lenet':
        model = LeNet(kwargs['num_classes'])#.cuda()
        return model
    elif backbone_name.lower() == 'resnet18':
        model = resnet18(num_classes=kwargs['num_classes'])
        return model
    elif backbone_name.lower() == 'resnet50':
        model = resnet50(num_classes=kwargs['num_classes'])
        return model
    elif backbone_name.lower() == 'resnet18_imagenet':
        if "pretrain" in kwargs and kwargs["pretrain"]=="pretrain":
            model = resnet18_imagenet(pretrained=True)
        else:
            model = resnet18_imagenet(pretrained=False)
        return model
    elif backbone_name.lower() == 'resnet50_imagenet':
        if "pretrain" in kwargs and kwargs["pretrain"]=="pretrain":
            model = resnet50_imagenet(pretrained=True)
        else:
            model = resnet50_imagenet(pretrained=False)
        return model
    elif backbone_name.lower() == 'resnet18_v2':
        model = resnet18_v2(num_classes=kwargs['num_classes'])
        return model
    elif backbone_name.lower() == 'resnet34_v2':
        model = resnet34_v2(num_classes=kwargs['num_classes'])
        return model
    elif backbone_name.lower() == 'resnet50_v2':
        model = resnet50_v2(num_classes=kwargs['num_classes'])
        return model
    else:
        logging.error("[%s] is not supported"%backbone_name)
        raise NotImplementedError("[%s] is not supported"%backbone_name)