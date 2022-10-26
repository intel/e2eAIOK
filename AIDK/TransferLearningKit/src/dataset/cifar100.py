import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from transformers import ViTFeatureExtractor

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

def get_transfrom(transform_opt,test=False):
    transform_list = []
    if not test:
        if "RandomCrop" in transform_opt["random"]:
            transform_list.append(transforms.RandomCrop(32, padding=4))
        if "RandomHorizontalFlip" in transform_opt["random"]:
            transform_list.append(transforms.RandomHorizontalFlip())
        if "RandomRotation" in transform_opt["random"]:
            transform_list.append(transforms.RandomRotation(15))
    if transform_opt["size"] > 32:
        transform_list.append(transforms.Resize(transform_opt["size"]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(transform_opt["norm_mean"], transform_opt["norm_std"]))
    transform_com = transforms.Compose(transform_list)
    return transform_com

def build_transfrom(cfg):
    ####For Train
    train_transform_dict_resnet = {
        "random": ["RandomCrop", "RandomHorizontalFlip","RandomRotation"],
        "size": 32,
        "norm_mean": CIFAR100_TRAIN_MEAN,
        "norm_std": CIFAR100_TRAIN_STD
    }
    train_transform_dict_denascnn= {
        "random": ["RandomCrop", "RandomHorizontalFlip"],
        "size": 32,
        "norm_mean": CIFAR100_TRAIN_MEAN,
        "norm_std": CIFAR100_TRAIN_STD
    }
    train_transform_dict_pretrainI21k= {
        "random": ["RandomCrop", "RandomHorizontalFlip"],
        "size": 112,
        "norm_mean": CIFAR100_TRAIN_MEAN,
        "norm_std": CIFAR100_TRAIN_STD
    }
    train_transform_dict_vit= {
        "random": ["RandomCrop", "RandomHorizontalFlip"],
        "size": 224,
        "norm_mean": IMAGE_MEAN,
        "norm_std": IMAGE_STD
    }
    train_transform_dict_vit_train= {
        "random": ["RandomCrop", "RandomHorizontalFlip"],
        "size": 32,
        "norm_mean": IMAGE_MEAN,
        "norm_std": IMAGE_STD
    }

    train_transform_dict = {
        "default": train_transform_dict_resnet,
        "resnet": train_transform_dict_resnet, 
        "denascnn": train_transform_dict_denascnn, # prefer: bin
        "pretrainI21k":train_transform_dict_pretrainI21k, # prefer: pretrain112
        "vit": train_transform_dict_vit, #prefer: vit_hg
        "vit_train": train_transform_dict_vit_train, #prefer: vit_hg_train
    }
    train_transform = get_transfrom(train_transform_dict[cfg.dataset.train_transform])

    ######For Test
    test_transform_dict_resnet= {
        "size": 32,
        "norm_mean": CIFAR100_TRAIN_MEAN,
        "norm_std": CIFAR100_TRAIN_STD
    }
    test_transform_dict_pretrainI21k= {
        "size": 112,
        "norm_mean": CIFAR100_TRAIN_MEAN,
        "norm_std": CIFAR100_TRAIN_STD
    }
    test_transform_dict_vit= {
        "size": 224,
        "norm_mean": IMAGE_MEAN,
        "norm_std": IMAGE_STD
    }
    test_transform_dict_vit_train= {
        "size": 32,
        "norm_mean": IMAGE_MEAN,
        "norm_std": IMAGE_STD
    }

    test_transform_dict = {
        "default": test_transform_dict_resnet,
        "resnet": test_transform_dict_resnet,  # prefer: NONE
        "pretrainI21k":test_transform_dict_pretrainI21k,  # prefer: pretrain112
        "vit": test_transform_dict_vit,  #prefer: vit_hg
        "vit_train":test_transform_dict_vit_train  #prefer: vit_hg_train
    }
    test_transform = get_transfrom(test_transform_dict[cfg.dataset.test_transform], test=True)

    return train_transform, test_transform

def get_data_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_cifar100_dataset(cfg):
    data_folder = get_data_folder(cfg.dataset.path)
    train_transform, test_transform = build_transfrom(cfg)
 
    train_set = datasets.CIFAR100(root=data_folder, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_folder, train=False, download=True, transform=test_transform)

    num_classes = 100

    return train_set, test_set, test_set, num_classes

