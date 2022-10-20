#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import random_split
from .utils.image_list import ImageList
from torchvision import transforms

def get_usps_vs_minist_dataset(cfg):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])
    ])
    num_classes = 10
    test_dataset = ImageList(os.path.join(cfg.dataset.path,"MNIST"), 
                                open(os.path.join(cfg.dataset.path,"MNIST/mnist_test.txt")).readlines(),
                                transform, 'L')
    train_dataset = ImageList(os.path.join(cfg.dataset.path,"MNIST"), 
                                open(os.path.join(cfg.dataset.path,"MNIST/mnist_train.txt")).readlines(),
                                transform, 'L')
    test_len = len(test_dataset)
    test_dataset, validation_dataset = random_split(test_dataset,[test_len // 2, test_len - test_len // 2])
    
    return train_dataset, validation_dataset,  test_dataset, num_classes
    