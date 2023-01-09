import os
import numpy as np

from torchvision import transforms
from e2eAIOK.common.trainer.data.cv.data_builder_cifar import DataBuilderCIFAR

CIFAR_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_TRAIN_STD = (0.2023, 0.1994, 0.2010)
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

class DataBuilderCIFARMA(DataBuilderCIFAR):
    def __init__(self, cfg):
        super().__init__(cfg)
        train_transform_dict, test_transform_dict = self.define_transform_dict()
        self.train_transform_dict = train_transform_dict
        self.test_transform_dict = test_transform_dict

    def define_transform_dict(self):
        ####For Train
        train_transform_dict_default= {
            "random": ["RandomCrop", "RandomHorizontalFlip"],
            "norm_mean": CIFAR_TRAIN_MEAN,
            "norm_std": CIFAR_TRAIN_STD
        }
        train_transform_dict_resnet = {
            "random": ["RandomCrop", "RandomHorizontalFlip","RandomRotation"],
            "norm_mean": CIFAR_TRAIN_MEAN,
            "norm_std": CIFAR_TRAIN_STD
        }
        train_transform_dict_vit= {
            "random": ["RandomCrop", "RandomHorizontalFlip"],
            "norm_mean": IMAGE_MEAN,
            "norm_std": IMAGE_STD
        }

        train_transform_dict = {
            "default": train_transform_dict_default,
            "resnet": train_transform_dict_resnet, 
            "vit": train_transform_dict_vit
        }
        
        ######For Test
        test_transform_dict_default= {
            "norm_mean": CIFAR_TRAIN_MEAN,
            "norm_std": CIFAR_TRAIN_STD
        }
        test_transform_dict_vit= {
            "norm_mean": IMAGE_MEAN,
            "norm_std": IMAGE_STD
        }

        test_transform_dict = {
            "default": test_transform_dict_default,
            "vit": test_transform_dict_vit
        }

        return train_transform_dict, test_transform_dict

    def get_transfrom(self, transform_opt, test=False):
        transform_list = []
        if not test:
            if "RandomCrop" in transform_opt["random"]:
                transform_list.append(transforms.RandomCrop(32, padding=4))
            if "RandomHorizontalFlip" in transform_opt["random"]:
                transform_list.append(transforms.RandomHorizontalFlip())
            if "RandomRotation" in transform_opt["random"]:
                transform_list.append(transforms.RandomRotation(15))
        if self.cfg.input_size > 32:
            transform_list.append(transforms.Resize(self.cfg.input_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(transform_opt["norm_mean"], transform_opt["norm_std"]))
        transform_com = transforms.Compose(transform_list)
        return transform_com    

    def build_transform(self, is_train):
        if is_train:
            transform = self.get_transfrom(self.train_transform_dict[self.cfg.train_transform])
        else:
            transform = self.get_transfrom(self.test_transform_dict[self.cfg.test_transform], test=True)
        return transform