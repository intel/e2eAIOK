#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch

from .adversarial_base import AdversarialNetwork
import torch.nn as nn
import logging

class DANN(AdversarialNetwork):
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
        super(DANN, self).__init__(in_feature, hidden_size,dropout_rate,grl_coeff_alpha,grl_coeff_high,max_iter)

    def loss(self, discriminator_feature_source, backbone_output_source, backbone_label_source,
             discriminator_feature_target, backbone_output_target, backbone_label_target,
             tensorboard_writer = None):
        ''' discriminator loss function

        :param discriminator_feature_source: discriminator input feature of source
        :param backbone_output_source: backbone logit output of source
        :param backbone_label_source: backbone ground truth of source
        :param discriminator_feature_target: discriminator input feature of target
        :param backbone_output_target: backbone logit output of target
        :param backbone_label_target: backbone ground truth of target
        :param tensorboard_writer: tensorboard writer
        :return: loss
        '''
        prediction = self.forward(torch.concat([discriminator_feature_source,
                          discriminator_feature_target],dim=0)) # discriminator prediction, shape: [2*batchsize,1]
        if tensorboard_writer is not None:
            tensorboard_writer.add_histogram('DANN/prediction', prediction, self.iter_num)

        label = torch.concat([self.make_label(discriminator_feature_source.size(0),True),
                                            self.make_label(discriminator_feature_target.size(0), False)
                                            ],dim=0).view(-1,1) # discriminator label, shape: [2*batchsize,1]
        return nn.BCELoss()(prediction,label) # input should be 0~1
