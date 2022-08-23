import logging
import os
import time
import numpy as np
import torch

from pt.pytorch_trainer import PytorchTrainer

def load_ckp(dir):
    model = torch.load(os.path.join(dir, 'model.pth'))
    optimizer = torch.load(os.path.join(dir, 'optimizer.pth'))
    return model, optimizer

def train(args, model, config):
    trainer = PytorchTrainer(args, model)
    metric = trainer.train(config['train_dataset'], config['eval_dataset'])
    return metric

def evaluate(args, model, config):
    pass