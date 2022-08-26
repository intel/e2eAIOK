import random
import sys
import argparse
import torch
import yaml
import numpy as np
from utils import update_config
from model.cv.vit_trainer import ViTTrainer


def parse_args(args):
    parser = argparse.ArgumentParser('Best module training............')
    parser.add_argument('--domain', type=str, default=None, choices=['cnn','vit'], help='DE-NAS model domain')
    parser.add_argument('--conf', type=str, default=None, help='DE-NAS training conf file')
    parser.add_argument('--dist-train', action='store_true', default=False, help='Enabling distributed traing')

    train_args, model_args = parser.parse_known_args(args)
    update_config(model_args, train_args.conf)
    return train_args, model_args

def main(train_args, model_args):
    """The unified trainer for DE-NAS to load the searched best model structure, conduct training and generate the compact model. 
    :param train_args: the overall arguments
    :param model_args: specific model arguments
    """
    if train_args.domain == 'vit':
        if train_args.dist_train == True:
            model_args = model_args[1:]
        vit_trainer = ViTTrainer(model_args)
        vit_trainer.fit()
    else:
        pass

if __name__ == '__main__':
    train_args, model_args  = parse_args(sys.argv[1:])
    main(train_args, model_args)