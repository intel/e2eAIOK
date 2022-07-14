#!/usr/bin/python
# -*- coding: utf-8 -*-
from .adversarial.CDAN import CDAN
from .adversarial.DANN import DANN
import logging

def createDiscriminator(discriminator_name,**kwargs):
    ''' create discriminator by name

    :param discriminator_name: discriminator name
    :param kwargs: kwargs to create discriminator
    :return: a discriminator model
    '''
    if discriminator_name == 'DANN':
        return DANN(in_feature=int(kwargs['input_size']),
                    hidden_size=int(kwargs['hidden_size']),
                    dropout_rate=float(kwargs['dropout']),
                    grl_coeff_alpha=float(kwargs['grl_coeff_alpha']),
                    grl_coeff_high=float(kwargs['grl_coeff_high']),
                    max_iter=int(kwargs['max_iter']))
    elif discriminator_name == 'CDAN':
        return CDAN(in_feature=int(kwargs['input_size']),
                    hidden_size=int(kwargs['hidden_size']),
                    dropout_rate=float(kwargs['dropout']),
                    grl_coeff_alpha=float(kwargs['grl_coeff_alpha']),
                    grl_coeff_high=float(kwargs['grl_coeff_high']),
                    max_iter=int(kwargs['max_iter']),
                    backbone_output_size = int(kwargs['backbone_output_size']),
                    enable_random_layer= int(kwargs['enable_random_layer']) > 0,
                    enable_entropy_weight= int(kwargs['enable_entropy_weight']) > 0)
    else:
        logging.error("[%s] is not supported" % discriminator_name)
        raise NotImplementedError("[%s] is not supported" % discriminator_name)