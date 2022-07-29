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