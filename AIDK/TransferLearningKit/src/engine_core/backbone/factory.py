#!/usr/bin/python
# -*- coding: utf-8 -*-
from .lenet import LeNet
from .resnet import resnet18
import logging

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
        pretrained_path = kwargs['pretrained_path'] if 'pretrained_path' in kwargs else None
        pretrained_layer_pattern = kwargs['pretrained_layer_pattern'] if 'pretrained_layer_pattern' in kwargs else None
        model = resnet18(num_classes=kwargs['num_classes'],pretrained_path=pretrained_path,
                         pretrained_layer_pattern=pretrained_layer_pattern)
        return model
    else:
        logging.error("[%s] is not supported"%backbone_name)
        raise NotImplementedError("[%s] is not supported"%backbone_name)
