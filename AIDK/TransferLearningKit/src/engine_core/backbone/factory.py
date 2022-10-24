#!/usr/bin/python
# -*- coding: utf-8 -*-
from .lenet import LeNet
from .resnet_imagenet import resnet18 as resnet18_imagenet
from .resnet_imagenet import resnet50 as resnet50_imagenet
from .resnetv2 import ResNet18 as resnet18_v2
from .resnetv2 import ResNet34 as resnet34_v2
from .resnetv2 import ResNet50 as resnet50_v2
from .resnet_cifar import ResNet18 as resnet18_cifar
from .resnet_cifar import ResNet50 as resnet50_cifar
from .utils import initWeights
import logging
import torch
import timm
import sys, os
import argparse

def createBackbone(backbone_name, num_classes, pretrain = None, **kwargs):
    ''' create backbone by name

    :param backbone_name: backbone name
    :param num_classes: num of classes
    :param pretrain: pretrained model
    :param kwargs: kwargs to create backbone
    :return: a backbone model
    '''
    backbone_name = backbone_name.lower()
    pretrained_flag = isinstance(pretrain,bool) and pretrain == True

    if backbone_name == 'lenet':
        model = LeNet(num_classes)#.cuda()
    elif backbone_name == 'resnet18_imagenet':
        model = resnet18_imagenet(pretrained=pretrained_flag)
    elif backbone_name == 'resnet50_imagenet':
        model = resnet50_imagenet(pretrained=pretrained_flag)
    elif backbone_name == 'resnet18_v2':
        model = resnet18_v2(num_classes=num_classes)
    elif backbone_name == 'resnet34_v2':
        model = resnet34_v2(num_classes=num_classes)
    elif backbone_name == 'resnet50_v2':
        model = resnet50_v2(num_classes=num_classes)
    elif backbone_name == "resnet18_cifar":
        model = resnet18_cifar(num_classes=num_classes)
    elif backbone_name == "resnet50_cifar":
        model = resnet50_cifar(num_classes=num_classes)
    elif backbone_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=pretrained_flag, num_classes=num_classes)
    elif backbone_name == 'mobilenet_v3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained_flag, num_classes=num_classes)
    elif backbone_name == 'vit_base':
        model = timm.create_model('vit_base_patch16_224_miil', pretrained=pretrained_flag, num_classes=num_classes)
    elif backbone_name == "denas_cnn":
        curdir = os.path.abspath(os.path.dirname(__file__))
        sys.path.append(os.path.join(curdir,"DeNas"))
        from .DeNas.trainer.model.cv.cnn_model_builder import CNNModelBuilder
        args_dict = {
            "num_classes": num_classes,
            "best_model_structure":os.path.join(curdir,"DeNas/best_model_structure.txt")
        }
        args = argparse.Namespace(**args_dict)
        model_builder = CNNModelBuilder(args)
        with open(args.best_model_structure, 'r') as f:
            arch = f.readlines()[-1]
        model = model_builder.create_model(arch)
    elif backbone_name == "vit_base_224_in21k_ft_cifar100":
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')
    else:
        logging.error("[%s] is not supported"%backbone_name)
        raise NotImplementedError("[%s] is not supported"%backbone_name)

    if not pretrained_flag:
        model.apply(initWeights)
        if pretrain:
            if not os.path.exists(pretrain):
                raise RuntimeError(f"Can not find {pretrain}!")
            print(f"load pretrained model at {pretrain}")
            logging.info(f"load pretrained model at {pretrain}")
            state_dict = torch.load(pretrain,map_location=torch.device('cpu'))
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            model.load_state_dict(state_dict, strict=True)
    return model