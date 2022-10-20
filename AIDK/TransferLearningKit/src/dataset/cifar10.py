import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .utils.dataset_wrapper import DatasetWrapper

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

def get_transfrom(transform_opt,test=False):
    transform_list = []
    if not test:
        if "RandomCrop" in transform_opt["random"]:
            transform_list.append(transforms.RandomCrop(32, padding=4))
        if "RandomHorizontalFlip" in transform_opt["random"]:
            transform_list.append(transforms.RandomHorizontalFlip())
    if transform_opt["size"] > 32:
        transform_list.append(transforms.Resize(transform_opt["size"]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD))
    transform_com = transforms.Compose(transform_list)
    return transform_com

def build_transfrom(cfg):
    ####For Train
    train_transform_dict_denascnn = {
        "random": ["RandomCrop", "RandomHorizontalFlip"],
        "size": 32
    }
    train_transform_dict_pretrainI21k = {
        "random": ["RandomCrop", "RandomHorizontalFlip"],
        "size": 112
    }

    train_transform_dict = {
        "default": train_transform_dict_denascnn,
        "denascnn": train_transform_dict_denascnn,
        "pretrainI21k":train_transform_dict_pretrainI21k,
    }
    train_transform = get_transfrom(train_transform_dict[cfg.dataset.train_transform])

    ####For Test
    test_transform_dict_resnet= {
        "size": 32,
    }
    test_transform_dict_pretrainI21k= {
        "size": 112,
    }

    test_transform_dict = {
        "default": test_transform_dict_resnet,
        "resnet": test_transform_dict_resnet,
        "pretrainI21k":test_transform_dict_pretrainI21k
    }
    test_transform = get_transfrom(test_transform_dict[cfg.dataset.test_transform], test=True)

    return train_transform, test_transform

def get_data_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_cifar10_dataset(cfg):
    data_folder = get_data_folder(cfg.dataset.path)

    train_transform, test_transform = build_transfrom(cfg)

    train_set = datasets.CIFAR10(root=data_folder, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_folder, train=False, download=True, transform=test_transform)

    num_classes = 10

    if cfg.distiller.save_logits or cfg.distiller.use_saved_logits or cfg.distiller.check_logits:
        train_set = DatasetWrapper(train_set,
                                    logits_path=cfg.distiller.logits_path,
                                    num_classes = num_classes,
                                    topk=cfg.distiller.logits_topk,
                                    write=cfg.distiller.save_logits,
                                    )

    return train_set, test_set, test_set, num_classes

