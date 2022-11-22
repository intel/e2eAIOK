import os
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def build_dataset(is_train, cfg):
    transform = build_transform(is_train,cfg)

    if cfg.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(cfg.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif cfg.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(cfg.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100    
    return dataset, nb_classes

def build_transform(is_train, cfg):
    resize_im = cfg.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                cfg.input_size, padding=4)
        return transform
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform
