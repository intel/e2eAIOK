#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import random_split
from dataset.image_list import ImageList
from torchvision import transforms

def get_usps_vs_minist_dataloaders(path, batch_size,val_batch_size,num_workers,is_distributed=False):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageList(os.path.join(path,"MNIST"), 
                                open(os.path.join(path,"MNIST/mnist_test.txt")).readlines(),
                                transform, 'L')
    train_dataset = ImageList(os.path.join(path,"MNIST"), 
                                open(os.path.join(path,"MNIST/mnist_train.txt")).readlines(),
                                transform, 'L')
    test_len = len(test_dataset)
    test_dataset, validation_dataset = random_split(test_dataset,[test_len // 2, test_len - test_len // 2])
    
    num_data = len(train_dataset)

    if is_distributed:
        train_loader = torch.utils.data.DataLoader(train_dataset,  # only split train dataset
                                            batch_size=batch_size, shuffle=False,
                                            # shuffle is conflict with sampler
                                            num_workers=num_workers, drop_last=True,
                                            sampler=DistributedSampler(train_dataset))
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset,
        batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    return train_loader,validate_loader,  test_loader, num_data
    