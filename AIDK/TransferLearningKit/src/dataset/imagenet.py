import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class ImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_imagenet_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform

def get_imagenet_test_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return test_transform

def get_imagenet_dataloaders(path, batch_size, val_batch_size, num_workers,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std)
    train_folder = os.path.join(path, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data

def get_imagenet_val_loader(path, val_batch_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    test_transform = get_imagenet_test_transform(mean, std)
    test_folder = os.path.join(path, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return test_loader
