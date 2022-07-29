#!/usr/bin/python
# -*- coding: utf-8 -*-
from .adversarial.cdan_adapter import CDANAdapter
from .adversarial.dann_adapter import DANNAdapter
import logging

def createAdapter(adapter_name, **kwargs):
    ''' create adapter by name

    :param adapter_name: adapter name
    :param kwargs: kwargs to create adapter
    :return: a adapter model
    '''
    if adapter_name == 'DANN':
        return DANNAdapter(in_feature=int(kwargs['input_size']),
                    hidden_size=int(kwargs['hidden_size']),
                    dropout_rate=float(kwargs['dropout']),
                    grl_coeff_alpha=float(kwargs['grl_coeff_alpha']),
                    grl_coeff_high=float(kwargs['grl_coeff_high']),
                    max_iter=int(kwargs['max_iter']))
    elif adapter_name == 'CDAN':
        return CDANAdapter(in_feature=int(kwargs['input_size']),
                    hidden_size=int(kwargs['hidden_size']),
                    dropout_rate=float(kwargs['dropout']),
                    grl_coeff_alpha=float(kwargs['grl_coeff_alpha']),
                    grl_coeff_high=float(kwargs['grl_coeff_high']),
                    max_iter=int(kwargs['max_iter']),
                    backbone_output_size = int(kwargs['backbone_output_size']),
                    enable_random_layer= int(kwargs['enable_random_layer']) > 0,
                    enable_entropy_weight= int(kwargs['enable_entropy_weight']) > 0)
    else:
        logging.error("[%s] is not supported" % adapter_name)
        raise NotImplementedError("[%s] is not supported" % adapter_name)