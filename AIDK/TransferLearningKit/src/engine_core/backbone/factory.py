#!/usr/bin/python
# -*- coding: utf-8 -*-
from .lenet import LeNet
import logging

def createBackbone(backbone_name,**kwargs):
    ''' create backbone by name

    :param backbone_name: backbone name
    :param kwargs: kwargs to create backbone
    :return: a backbone model
    '''
    if backbone_name.lower() == 'lenet':
        model = LeNet(kwargs['num_class'])#.cuda()
        return model
    else:
        logging.error("[%s] is not supported"%backbone_name)
        raise NotImplementedError("[%s] is not supported"%backbone_name)
