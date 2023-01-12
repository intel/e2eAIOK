#!/usr/bin/python
# -*- coding: utf-8 -*-
from e2eAIOK.common.trainer.model.cv.resnet_cifar import ResNet18 as resnet18_cifar
from e2eAIOK.common.trainer.model.cv.resnet_cifar import ResNet50 as resnet50_cifar
from e2eAIOK.common.trainer.model.cv.lenet import LeNet
from e2eAIOK.common.trainer.model.model_utils.model_utils import initWeights
from e2eAIOK.common.trainer.model.model_builder_cv import ModelBuilderCV
import logging
import torch
import timm
import sys, os
import argparse

def createBackbone(cfg, model_type, num_classes, initial_pretrain=False, pretrain=""):
    ''' create backbone by name 
    :param cfg: configurations
    :param num_classes: number of classes
    :param initial_pretrain: whether load default pretrain weights
    :pretrain: pretrain model path
    '''
    backbone_name = model_type.lower()
    if backbone_name == 'lenet':
        model = LeNet(num_classes)#.cuda()
    elif backbone_name == "resnet18_cifar":
        model = resnet18_cifar(num_classes=num_classes)
    elif backbone_name == "resnet50_cifar":
        model = resnet50_cifar(num_classes=num_classes)
    elif backbone_name == 'resnet18':
        model = timm.create_model('resnet18', pretrained=initial_pretrain, num_classes=num_classes)
    elif backbone_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=initial_pretrain, num_classes=num_classes)
    elif backbone_name == 'mobilenet_v3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=initial_pretrain, num_classes=num_classes)
    elif backbone_name == 'vit_base':
        model = timm.create_model('vit_base_patch16_224_miil', pretrained=initial_pretrain, num_classes=num_classes)
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
    elif backbone_name == "huggingface_vit_base_224_in21k_ft_cifar100":
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')
    elif backbone_name == "unet":
        from .unet.generic_UNet_DA import Generic_UNet_DA
        args_dict = {
            "threeD": True, 
            "input_channels": 1, 
            "base_num_features": 30, 
            "num_classes": num_classes, 
            "num_conv_per_stage": 2,  
            "pool_op_kernel_sizes": None,
            "conv_kernel_sizes": None
        }
        model = Generic_UNet_DA(**args_dict)
    else:
        logging.error("[%s] is not supported"%backbone_name)
        raise NotImplementedError("[%s] is not supported"%backbone_name)

    if (not initial_pretrain) and (not pretrain):
        model.apply(initWeights)

    modelbuilder = ModelBuilderCV(cfg, model)
    model = modelbuilder.create_model(pretrain)
       
    return model