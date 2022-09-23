import sys
import argparse
import numpy as np
import torch
import random
from utils import update_config
from model.cv.vit_trainer import ViTTrainer
from model.cv.cnn_trainer import CNNTrainer
from model.nlp.bert_trainer import BertTrainer
from model.asr.asr_trainer import ASRTrainer

def parse_args(args):
    parser = argparse.ArgumentParser('Best module training............')
    parser.add_argument('--domain', type=str, default=None, choices=['cnn','vit','bert','asr'], help='DE-NAS model domain')
    parser.add_argument('--conf', type=str, default=None, help='DE-NAS training conf file')
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for consistent training')

    train_args, model_args = parser.parse_known_args(args)
    if train_args.conf is not None:
        update_config(model_args, train_args.conf)
    return train_args, model_args

def main(train_args, model_args):
    """The unified trainer for DE-NAS to load the searched best model structure, conduct training and generate the compact model. 
    :param train_args: the overall arguments
    :param model_args: specific model arguments
    """
    if train_args.random_seed:
        random.seed(train_args.random_seed)
        np.random.seed(train_args.random_seed)
        torch.manual_seed(train_args.random_seed)
        
    if train_args.domain == 'vit':
        vit_trainer = ViTTrainer(model_args)
        vit_trainer.fit()
    elif train_args.domain == 'bert':
        bert_trainer = BertTrainer(model_args)
        bert_trainer.fit()
    elif train_args.domain == 'asr':
        trainer = ASRTrainer(model_args)
        trainer.fit()
    elif train_args.domain == 'cnn':
        cnn_trainer = CNNTrainer(model_args)
        cnn_trainer.fit()
    else:
        pass

if __name__ == '__main__':
    train_args, model_args  = parse_args(sys.argv[1:])
    main(train_args, model_args)