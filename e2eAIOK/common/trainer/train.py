import sys
import argparse
import numpy as np
import torch
import random
import yaml
from easydict import EasyDict as edict
from torch_trainer import TorchTrainer 
from e2eAIOK.DeNas.asr.asr_trainer import ASRTrainer

def parse_args(args):
    parser = argparse.ArgumentParser('Torch model training or evluation............')
    parser.add_argument('--domain', type=str, default=None, choices=['cnn','vit','bert','asr'], help='training model domain')
    parser.add_argument('--conf', type=str, default=None, help='training or evluation conf file')
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for consistent training')
    train_args = parser.parse_args(args)
    return train_args

def main(args):
    """The unified trainer for DE-NAS to load the searched best model structure, conduct training and generate the compact model. 
    :param train_args: the overall arguments
    :param model_args: specific model arguments
    """
    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
    with open(args.conf) as f:
        cfg = edict(yaml.safe_load(f))

    if args.domain in ['cnn','vit']:
        trainer = TorchTrainer(cfg)
    elif args.domain == 'bert':
        ## TODO 
        trainer = BertTrainer(cfg)
    elif args.domain == 'asr':
        trainer = ASRTrainer(cfg)
    else:
        raise RuntimeError(f"Domain {args.domain} is not supported")
    trainer.fit()

if __name__ == '__main__':
    args  = parse_args(sys.argv[1:])
    main(args)