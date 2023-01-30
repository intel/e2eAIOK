#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch

from .adversarial_adapter import AdversarialAdapter
import torch.nn as nn
import logging

class DANNAdapter(AdversarialAdapter):
    ''' DANN, see paper: Domain-Adversarial Training of Neural Networks

    '''
    def __init__(self,in_feature, hidden_size,dropout_rate,grl_coeff_alpha,grl_coeff_high,max_iter):
        ''' init method.

        :param in_feature: adversarial network input
        :param hidden_size: hidden size
        :param dropout_rate: dropout rate
        :param grl_coeff_alpha: GradientReverseLayer alpha argument to calculate coef
        :param grl_coeff_high: GradientReverseLayer coef high
        :param max_iter: max iter for one epoch
        '''
        super(DANNAdapter, self).__init__(in_feature, hidden_size,dropout_rate,grl_coeff_alpha,grl_coeff_high,max_iter)

    def loss(self,output_prob,label,**kwargs):
        ''' adapter loss function

        :param output_prob: output probability
        :param label: ground truth
        :param kwargs: kwargs
        :return: loss
        '''
        return nn.BCELoss()(output_prob,label) # input should be 0~1
