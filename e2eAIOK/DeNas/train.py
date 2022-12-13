import sys
import os
from pathlib import Path
import argparse
import numpy as np
import torch
import random
import yaml
from easydict import EasyDict as edict
import sentencepiece as sp
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer 
import e2eAIOK.common.trainer.utils.utils as utils
from e2eAIOK.DeNas.asr.model_builder_denas_asr import ModelBuilderASRDeNas
from e2eAIOK.DeNas.cv.model_builder_denas_cv import ModelBuilderCVDeNas
from e2eAIOK.DeNas.nlp.model_builder_denas_nlp import ModelBuilderNLPDeNas
from e2eAIOK.common.trainer.data.asr.data_builder_librispeech import DataBuilderLibriSpeech
from e2eAIOK.common.trainer.data.cv.data_builder_cifar import DataBuilderCIFAR
from e2eAIOK.common.trainer.data.nlp.data_builder_squad import DataBuilderSQuAD
from e2eAIOK.DeNas.asr.asr_trainer import ASRTrainer
from e2eAIOK.DeNas.asr.trainer.schedulers import NoamScheduler
from e2eAIOK.DeNas.asr.trainer.losses import ctc_loss, kldiv_loss
from e2eAIOK.DeNas.asr.utils.metric_stats import ErrorRateStats
from e2eAIOK.DeNas.cv.cv_trainer import CVTrainer
from e2eAIOK.DeNas.nlp.utils import bert_create_optimizer, bert_create_criterion, bert_create_scheduler, bert_create_metric
from e2eAIOK.DeNas.nlp.bert_trainer import BERTTrainer


def parse_args(args):
    parser = argparse.ArgumentParser('Torch model training or evluation............')
    parser.add_argument('--domain', type=str, default=None, choices=['cnn','vit','bert','asr'], help='training model domain')
    parser.add_argument('--conf', type=str, default=None, help='training or evluation conf file')
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for consistent training')
    train_args = parser.parse_args(args)
    return train_args

def main(args):
    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    root_dir = Path(os.getcwd()).parent.parent
    conf_file = os.path.join(root_dir, args.conf)
    with open(conf_file) as f:
        cfg = edict(yaml.safe_load(f))

    ext_dist.init_distributed(backend=cfg.dist_backend)

    if args.domain in ['cnn','vit']:
        model = ModelBuilderCVDeNas(cfg).create_model()
        train_dataloader, eval_dataloader = DataBuilderCIFAR(cfg).get_dataloader()
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = CVTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
    elif args.domain == 'bert':
        model = ModelBuilderNLPDeNas(cfg).create_model()
        train_dataloader, eval_dataloader, other_data = DataBuilderSQuAD(cfg).get_dataloader()
        cfg.num_train_steps = len(train_dataloader)
        optimizer = bert_create_optimizer(model, cfg)
        criterion = bert_create_criterion(cfg)
        scheduler = bert_create_scheduler(cfg)
        metric = bert_create_metric(cfg)
        trainer = BERTTrainer(cfg, model, train_dataloader, eval_dataloader, other_data, optimizer, criterion, scheduler, metric)
    elif args.domain == 'asr':
        model = ModelBuilderASRDeNas(cfg).create_model()
        tokenizer = sp.SentencePieceProcessor()
        train_dataloader, eval_dataloader = DataBuilderLibriSpeech(cfg, tokenizer).get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)
        criterion = {"ctc_loss": ctc_loss, "seq_loss": kldiv_loss}
        scheduler = NoamScheduler(lr_initial=cfg["lr_adam"], n_warmup_steps=cfg["n_warmup_steps"])
        metric = ErrorRateStats()
        trainer = ASRTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric, tokenizer)
    else:
        raise RuntimeError(f"Domain {args.domain} is not supported")
    trainer.fit()

if __name__ == '__main__':
    args  = parse_args(sys.argv[1:])
    main(args)