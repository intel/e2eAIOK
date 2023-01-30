#!/usr/bin/python
# -*- coding: utf-8 -*-
from .adversarial.cdan_adapter import CDANAdapter
from .adversarial.dann_adapter import DANNAdapter
from .adversarial.DA_Loss import CACDomainAdversarialLoss
import logging

def createAdapter(adapter_name, **kwargs):
    ''' create adapter by name

    :param adapter_name: adapter name
    :param kwargs: kwargs to create adapter
    :return: a adapter model
    '''
    if adapter_name == 'DANN':
        return DANNAdapter(**kwargs)
    elif adapter_name == 'CDAN':
        return CDANAdapter(**kwargs)
    elif adapter_name == "CAC_UNet":
        return CACDomainAdversarialLoss(**kwargs)
    else:
        logging.error("[%s] is not supported" % adapter_name)
        raise NotImplementedError("[%s] is not supported" % adapter_name)