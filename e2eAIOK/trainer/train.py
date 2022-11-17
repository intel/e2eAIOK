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
    parser.add_argument('--train_mode', type=str, default="train",choices=['train','eval'], help='train mode or evaluate mode')
    parser.add_argument('--train_epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--eval_epochs', type=int, default=1, help='evluate between how many training epochs')

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
        vit_trainer.fit(train_args)
    elif train_args.domain == 'bert':
        bert_trainer = BertTrainer(model_args)
        bert_trainer.fit(train_args)
    elif train_args.domain == 'asr':
        asr_trainer = ASRTrainer(model_args)
        asr_trainer.fit(train_args)
    elif train_args.domain == 'cnn':
        cnn_trainer = CNNTrainer(model_args)
        cnn_trainer.fit(train_args)
    else:
        raise RuntimeError(f"Domain {train_args.domain} is not supported")

if __name__ == '__main__':
    train_args, model_args  = parse_args(sys.argv[1:])
    main(train_args, model_args)