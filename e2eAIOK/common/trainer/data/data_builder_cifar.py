import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.common.trainer.data.data_builder_cv import DataBuilderCV

class DataBuilderCIFAR(DataBuilderCV):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def prepare_dataset(self):
        """
            prepare CV related dataset
        """
        if self.cfg.data_set in ["CIFAR10","CIFAR100"]:
            dataset_train = self.build_dataset(is_train=True)
            dataset_val = self.build_dataset(is_train=False)
        else:
            raise RuntimeError(f"dataset {self.cfg.data_set} not supported")
        return dataset_train, dataset_val
    
    def build_dataset(self, is_train = True):
        transform = self.build_transform(is_train)
        if self.cfg.data_set == 'CIFAR10':
            dataset = datasets.CIFAR10(self.cfg.data_path, train=is_train, transform=transform, download=True)
        elif self.cfg.data_set == 'CIFAR100':
            dataset = datasets.CIFAR100(self.cfg.data_path, train=is_train, transform=transform, download=True)
        return dataset
    def build_transform(self, is_train):
        resize_im = self.cfg.input_size > 32
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
                    self.cfg.input_size, padding=4)
            return transform
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        return transform