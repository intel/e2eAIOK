import os
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist


from e2eAIOK.common.trainer.data_builder import DataBuilder

class DataBuilderCV(DataBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def prepare_dataset(self):
        """
            prepare CV related dataset
        """
        if self.cfg.data_set in ["CIFAR10","CIFAR100"]:
            dataset_train = self.build_dataset(is_train=True, cfg=self.cfg)
            dataset_val = self.build_dataset(is_train=False, cfg=self.cfg)
        else:
            raise RuntimeError(f"dataset {self.cfg.data_set} not supported")
        return dataset_train, dataset_val
    
    def build_dataset(self):
        transform = build_transform(is_train,cfg)
        if cfg.data_set == 'CIFAR10':
            dataset = datasets.CIFAR10(cfg.data_path, train=is_train, transform=transform, download=True)
        elif cfg.data_set == 'CIFAR100':
            dataset = datasets.CIFAR100(cfg.data_path, train=is_train, transform=transform, download=True)
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
