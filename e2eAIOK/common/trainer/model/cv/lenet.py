#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import logging

class LeNet(nn.Module):
    ''' LeNet backbone

    '''
    def __init__(self,num_classes):
        ''' init method

        :param num_classes: num classes
        '''
        super(LeNet, self).__init__()
        self.num_classes = num_classes
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
        self.classifier = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        y = self.classifier(x)
        return y