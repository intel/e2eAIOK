#!/usr/bin/python
# -*- coding: utf-8 -*-
from PIL import Image
import torch

def rgb_loader(path):
    ''' load rgb image

    :param path: image path
    :return: image data
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    ''' load grayscale image

    :param path:  image path
    :return: image data
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class DataLoaderCollater:
    ''' collate_fn used in DataLoader

    '''
    def __init__(self,tramsform):
        ''' init method

        :param tramsform: a transform applied on every sample
        '''
        self._tramsform = tramsform

    def __call__(self, sample_list):
        ''' call function

        :param sample_list: a list of samples, [(sample, label),...]
        :return: transformed samples
        '''
        features,labels = zip(*sample_list)
        if self._tramsform is None:
            torch.stack(features)
        else:
            features = torch.stack([self._tramsform(i) for i in features])
        labels = torch.tensor(labels)
        return (features,labels)