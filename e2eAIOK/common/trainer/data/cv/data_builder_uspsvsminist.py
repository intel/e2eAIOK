#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import random_split
from e2eAIOK.common.trainer.data.data_utils.image_list import ImageList
from torchvision import transforms
from e2eAIOK.common.trainer.data.data_builder_cv import DataBuilderCV

class DataBuilderUSPSMinist(DataBuilderCV):
    def __init__(self, cfg):
        super().__init__(cfg)

    def prepare_dataset(self):
        """
            prepare USPS_vs_Minist dataset
        """
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            # transforms.Normalize(
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225])
        ])
        test_dataset = ImageList(os.path.join(self.cfg.data_path,"MNIST"), 
                                    open(os.path.join(self.cfg.data_path,"MNIST/mnist_test.txt")).readlines(),
                                    transform, 'L')
        train_dataset = ImageList(os.path.join(self.cfg.data_path,"MNIST"), 
                                    open(os.path.join(self.cfg.data_path,"MNIST/mnist_train.txt")).readlines(),
                                    transform, 'L')
        test_len = len(test_dataset)
        test_dataset, validation_dataset = random_split(test_dataset,[test_len // 2, test_len - test_len // 2])
        self.dataset_train = train_dataset
        self.dataset_val = validation_dataset
        self.dataset_test = test_dataset
    