#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/28/2022 10:55 AM

import torch.nn as nn

class BasicDistiller(nn.Module):
    ''' BasicDistiller

    '''
    def __init__(self,pretrained_model,is_frozen):
        ''' Init method.

        :param pretrained_model: the pretrained model as teacher
        :param is_frozen: whether frozen teacher when training
        '''
        super(BasicDistiller, self).__init__()
        self.pretrained_model = pretrained_model
        self._is_frozen = is_frozen
        if is_frozen:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.pretrained_model(x)

    def loss(self,teacher_logits,student_logits):
        return nn.MSELoss(teacher_logits,student_logits)

