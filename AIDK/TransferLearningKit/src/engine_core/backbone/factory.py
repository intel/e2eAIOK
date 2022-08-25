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
    backbone_name = backbone_name.lower()

    if backbone_name == 'lenet':
        model = LeNet(kwargs['num_classes'])#.cuda()
    elif backbone_name == 'resnet18':
        model = resnet18(num_classes=kwargs['num_classes'])
    elif backbone_name == 'resnet50':
        model = resnet50(num_classes=kwargs['num_classes'])
    elif backbone_name == 'resnet18_imagenet':
        if "pretrain" in kwargs and kwargs["pretrain"]=="pretrain":
            model = resnet18_imagenet(pretrained=True)
        else:
            model = resnet18_imagenet(pretrained=False)
    elif backbone_name == 'resnet50_imagenet':
        if "pretrain" in kwargs and kwargs["pretrain"]=="pretrain":
            model = resnet50_imagenet(pretrained=True)
        else:
            model = resnet50_imagenet(pretrained=False)
    elif backbone_name == 'resnet18_v2':
        model = resnet18_v2(num_classes=kwargs['num_classes'])
    elif backbone_name == 'resnet34_v2':
        model = resnet34_v2(num_classes=kwargs['num_classes'])
    elif backbone_name == 'resnet50_v2':
        model = resnet50_v2(num_classes=kwargs['num_classes'])
    else:
        logging.error("[%s] is not supported"%backbone_name)
        raise NotImplementedError("[%s] is not supported"%backbone_name)
    return model