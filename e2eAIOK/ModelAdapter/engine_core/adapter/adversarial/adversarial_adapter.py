#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import logging
from .grl import GradientReverseLayer

class AdversarialAdapter(nn.Module):
    ''' AdversarialNetwork used for adversarial domain adaption

    '''
    def __init__(self, in_feature, hidden_size,dropout_rate,grl_coeff_alpha,grl_coeff_high,max_iter):
        ''' init method.

        :param in_feature: adversarial network input
        :param hidden_size: hidden size
        :param dropout_rate: dropout rate
        :param grl_coeff_alpha: GradientReverseLayer alpha argument to calculate coef
        :param grl_coeff_high: GradientReverseLayer coef high
        :param max_iter: max iter for one epoch
        '''
        super(AdversarialAdapter, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.ad_layer3 = nn.Linear(hidden_size//2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.grl = GradientReverseLayer(coeff_alpha=grl_coeff_alpha,coeff_high=grl_coeff_high,
                                        max_iter=max_iter,enable_step=self.training)
        self.iter_num = 0

    def forward(self, x,**kwargs):
        x = self.grl(x)
        # coeff = np.float(2.0 * self.grl_coeff_high / (1.0 + np.exp(-self.grl_coeff_alpha * self.iter_num / self.max_iter)) - self.grl_coeff_high)
        # if self.iter_num % 10 ==0:
        #     print(self.grl_coeff_alpha,self.grl_coeff_high,self.iter_num ,self.max_iter,coeff)
        # x.register_hook(lambda g: -g*coeff)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        y = self.sigmoid(x)

        if self.training:
            self.iter_num += 1
        return y

    @classmethod
    def make_label(cls,shape,is_source):
        ''' create adapter label. If is_source = True then label is 1, else 0.

        :param shape: label shape
        :param is_source: whether it is source
        :return: adapter label
        '''
        if is_source:
            return torch.ones(shape,dtype=torch.float)   #.cuda()
        else:
            return torch.zeros(shape, dtype=torch.float)  # .cuda()

    def loss(self,output_prob,label,**kwargs):
        ''' adapter loss function

        :param output_prob: output probability
        :param label: ground truth
        :param kwargs: kwargs
        :return: loss
        '''
        raise NotImplementedError("must implement loss function")