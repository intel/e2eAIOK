import logging
import re
import torch
import torch.nn as nn
import os

def initWeights(layer):
    ''' Initialize layer parameters

    :param layer: the layer to be initialized
    :return:
    '''
    classname = layer.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        logging.info("init layer [%s] with kaiming_uniform"% classname)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        logging.info("init layer [%s] with normal" % classname)
    elif classname.find('Linear') != -1:
        if hasattr(layer, "weight"):
            nn.init.xavier_normal_(layer.weight)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)
        logging.info("init layer [%s] with xavier_normal" % classname)
    else:
        logging.info("no init layer[%s]"%classname)
        # print("no init layer[%s]"%classname)    
