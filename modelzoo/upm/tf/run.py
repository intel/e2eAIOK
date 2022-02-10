import logging
import os
import time
import numpy as np

from tf.tensorflow_trainer import TensorflowTrainer

def train(args, model, config):
    trainer = TensorflowTrainer(args, model)
    metric = trainer.train(config['train_dataset'], config['eval_dataset'])
    return metric

def evaluate(args, model, config):
    pass