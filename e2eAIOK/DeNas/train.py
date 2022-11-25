import sys
import argparse
import numpy as np
import torch
import random
import yaml
from easydict import EasyDict as edict
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer 
import e2eAIOK.common.trainer.utils.utils as utils
from e2eAIOK.common.trainer.model.model_builder_asr import ModelBuilderASR
from e2eAIOK.common.trainer.model.model_builder_cv import ModelBuilderCV
from e2eAIOK.common.trainer.model.model_builder_nlp import ModelBuilderNLP
from e2eAIOK.common.trainer.data.data_builder_asr import DataBuilderASR
from e2eAIOK.common.trainer.data.data_builder_cv import DataBuilderCV
from e2eAIOK.common.trainer.data.data_builder_nlp import DataBuilderNLP

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
        # TODO
        model = ModelBuilderCV.create_model(cfg)
        train_dataloader, eval_dataloader = DataBuilderCV.get_dataloader(cfg)
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = CVTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
    elif args.domain == 'bert':
        ## TODO 
        model = ModelBuilderNLP.create_model(cfg)
        train_dataloader, eval_dataloader = DataBuilderNLP.get_dataloader(cfg)
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = BERTTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
    elif args.domain == 'asr':
        # TODO
        model = ModelBuilderASR.create_model(cfg)
        train_dataloader, eval_dataloader = DataBuilderASR.get_dataloader(cfg)
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = ASRTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
    else:
        raise RuntimeError(f"Domain {args.domain} is not supported")
    trainer.fit()

if __name__ == '__main__':
    args  = parse_args(sys.argv[1:])
    main(args)