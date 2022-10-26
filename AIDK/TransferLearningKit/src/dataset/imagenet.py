import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

class ImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_imagenet_train_transform():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform

def get_imagenet_test_transform():
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return test_transform

def get_imagenet_dataset(cfg):
    train_transform = get_imagenet_train_transform()
    train_folder = os.path.join(cfg.dataset.path, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    test_set, test_loader = get_imagenet_val_loader(cfg)

    num_classes = 1000
    
    return train_set, test_set, test_set, num_classes

def get_imagenet_val_loader(cfg):
    test_transform = get_imagenet_test_transform()
    test_folder = os.path.join(cfg.dataset.path, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=cfg.dataset.val.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    return test_set, test_loader
