#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import logging

class LeNet(nn.Module):
    ''' LeNet backbone

    '''
    def __init__(self,class_num):
        ''' init method

        :param class_num: class num
        '''
        super(LeNet, self).__init__()
        self.class_num = class_num
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(nn.Linear(50 * 4 * 4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, class_num)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        y = self.classifier(x)
        return y,x

    def loss(self,output_logit_source,label,tensorboard_writer = None):
        ''' loss function

        :param output_logit_source: model prediction (raw logit)
        :param label: ground truth
        :param tensorboard_writer: tensorboard writer
        :return:
        '''
        return nn.CrossEntropyLoss()(output_logit_source, label) # The `input` is expected to contain raw, unnormalized scores for each class.


