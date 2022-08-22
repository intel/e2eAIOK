#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/28/2022 8:39 AM

import logging
from .utils import adjust_learning_rate
import torch
import datetime
import torch.nn
from engine_core.transferrable_model import TransferrableModel
from dataset.composed_dataset import ComposedDataset
from torch.nn.parallel import DistributedDataParallel
import os

def unwrap_DDP(model):
    ''' unwarp a DDP model

    :param model: a model
    :return: the unwrapped model
    '''
    if type(model) is DistributedDataParallel:
        return model.module
    else:
        return model

def add_tensorboard_metric(tensorboard_writer,dataset_name,metric_values,cur_epoch=0,cur_step=0,epoch_steps=0,rank=-1):
    ''' add metric to tensorboard

    :param tensorboard_writer: a tensorboard writer
    :param dataset_name: dataset name (Train? Evaluation? Test?)
    :param metric_values: metric name and metric value
    :param cur_epoch: current epoch
    :param cur_step: current step
    :param epoch_steps : step num of one epoch
    :param rank: rank for distributed training (-1 for non-distributed training)
    :return:
    '''
    if dataset_name not in ['Train','Validation','Test']:
        raise RuntimeError("dataset_name (%s) must in 'Train','Validation','Test'"%dataset_name)

    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
    for (metric_name, metric_value) in metric_values.items():
        tensorboard_writer.add_scalar('{}/{}_{}'.format(metric_name,dataset_name,metric_name), metric_value, cur_epoch * epoch_steps + cur_step)

    metric_str = ";\t".join("{} = {:.4f}".format(metric_name, metric_value) for (metric_name, metric_value) in metric_values.items())
    if dataset_name == 'Train':
        out_str = '[{}] {} epoch({}) step ({}/{}) {}: {}'.format(dt, "rank(%s)"%rank if rank >=0 else "",
                                                                 cur_epoch,cur_step,epoch_steps,dataset_name,metric_str)
    else:
        out_str = '[{}] {} epoch({}) {}: {}'.format(dt,"rank(%s)"%rank if rank >=0 else "",
                                                    cur_epoch,dataset_name, metric_str)
    print(out_str)
    logging.info(out_str)
def trainEpoch(model, metric_fn_map, optimizer, train_dataloader, 
               tensorboard_writer,cur_epoch,epoch_steps,logging_interval,device,rank):
    ''' train one epoch

    :param model: the training model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param optimizer: the optimizer
    :param train_dataloader: train dataloader
    :param epoch_steps: how many steps of an epoch
    :param tensorboard_writer: tensorboard writer
    :param cur_epoch: current epoch
    :param logging_interval: logging interval durring training
    :param rank: rank for distributed training (-1 for non-distributed training)
    :return:
    '''
    model.train()  # set training flag
    for (cur_step,(data, label)) in enumerate(train_dataloader):
        # data, label = data.cuda(), label.cuda()
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            label = label.to(device)
        else:
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            label[0] = label[0].to(device)
            label[1] = label[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_value = unwrap_DDP(model).loss(output, label,cur_epoch)
        loss_value.backward()

        if cur_step % logging_interval == 0:
            unwrapped_model = unwrap_DDP(model)
            if isinstance(unwrapped_model,TransferrableModel): # for DDP
                metric_values = unwrapped_model.get_training_metrics(output, label, loss_value, metric_fn_map)
            else:
                metric_values = {"loss": loss_value}
                for (metric_name, metric_fn) in sorted(metric_fn_map.items()):
                    metric_value = metric_fn(output, label)
                    metric_values[metric_name] = metric_value
            add_tensorboard_metric(tensorboard_writer, 'Train', metric_values, cur_epoch,cur_step,epoch_steps,rank)

        optimizer.step()

def evaluateEpoch(model, metric_fn_map, dataloader,tensorboard_writer,
                  cur_epoch,epoch_steps,test_flag,device,rank):
    ''' evaluate epoch

    :param model: the evaluated model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param dataloader: dataloader
    :param tensorboard_writer : tensorboard writer
    :param cur_epoch : current epoch
    :param epoch_steps: steps per step
    :param test_flag : whether is test or validation
    :param rank: rank for distributed training (-1 for non-distributed training)
    :return: metric_value_maps
    '''
    datasetName = 'Test' if test_flag else 'Validation'

    with torch.no_grad():
        model.eval()  # set evaluating flag

        loss_value = 0
        metric_values = {}
        sample_num = 0
        #################### iterate on dataset ##############
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            batch_size = data.size(0)
            sample_num += batch_size
            loss_value += model.loss(output, label).item() * batch_size
            for (metric_name,metric_fn) in metric_fn_map.items():
                metric_value = metric_fn(output,label)
                if metric_name not in metric_values:
                    metric_values[metric_name] = metric_value * batch_size
                else:
                    metric_values[metric_name] += metric_value * batch_size
        ############## average ###################
        metric_values['loss'] = loss_value
        for metric_name in sorted(metric_values.keys()):
            metric_values[metric_name] /= sample_num
        add_tensorboard_metric(tensorboard_writer,datasetName,metric_values,cur_epoch,cur_step=0,
                               epoch_steps=epoch_steps,rank=rank)

        return metric_values



class Trainer:
    ''' Trainer

    '''
    def __init__(self, model, optimizer, early_stopping, validate_metric_fn_map,early_stop_metric,training_epochs,
                 tensorboard_writer,logging_interval,finetuner=None,pretrain=None,teacher_pretrain=None,
                 device='cpu',init_weight=True,rank=-1):
        ''' Init method

        :param model: the trained model
        :param optimizer: optimizer
        :param early_stopping: for early stopping
        :param validate_metric_fn_map: metric function map, which map metric name to metric function
        :param early_stop_metric: metric name for early stopping
        :param training_epochs: max training epochs
        :param tensorboard_writer: tensorboard writer
        :param logging_interval: training logging interval
        :param rank: rank for distributed training (-1 for non-distributed training)
        '''
        self._model = model
        self._optimizer = optimizer
        self._early_stopping = early_stopping
        self._validate_metric_fn_map = validate_metric_fn_map
        self._early_stop_metric = early_stop_metric
        self._training_epochs = training_epochs
        self._tensorboard_writer = tensorboard_writer
        self._logging_interval = logging_interval
        self._rank = rank
        self._best_metrics = 0.0
        self.device = device

        if early_stop_metric not in validate_metric_fn_map:
            raise RuntimeError("early stop metric [%s] not in validate_metric_fn_map keys [%s]"%(
                early_stop_metric,",".join(validate_metric_fn_map.keys())
            ))

        unwrap_DDP(self._model).init_weight(finetuner,pretrain,teacher_pretrain)

    def __str__(self):
        _str = "Trainer: model:%s\n"%self._model
        _str += "\toptimizer:%s\n"%self._optimizer
        _str += "\tearly_stopping:%s\n" % self._early_stopping
        _str += "\tvalidate_metric_fn_map:%s\n" % self._validate_metric_fn_map
        _str += "\t_early_stop_metric:%s\n" % self._early_stop_metric
        _str += "\t_training_epochs:%s\n" % self._training_epochs
        _str += "\ttensorboard_writer:%s\n" % self._tensorboard_writer
        _str += "\tlogging_interval:%s\n" % self._logging_interval
        _str += "\trank:%s\n" % self._rank
        return _str

    def train(self, train_dataloader,epoch_steps,valid_dataloader,cfg,model_dir,resume=False):
        ''' train function, and save the best trained model to cfg.EXPERIMENT.MODEL_SAVE

        :param train_dataloader: train dataloader
        :param epoch_steps: steps per epoch
        :param valid_dataloader: validation dataloader
        :param cfg: config settings
        :return:
        '''
        initial_epoch = 1
        if resume:
            # state = torch.load(cfg.MODEL.PRETRAIN, map_location=torch.device('cpu'))
            state = torch.load(os.path.join(cfg.EXPERIMENT.MODEL_SAVE, cfg.EXPERIMENT.PROJECT,cfg.EXPERIMENT.TAG,"latest.pth"), map_location=self.device)
            initial_epoch = state["epoch"] + 1
            unwrap_DDP(self._model).load_state_dict(state["model"])
            # self._optimizer.load_state_dict(state["optimizer"])
            self._best_metrics = state["best_metric"]
        for epoch in range(initial_epoch, self._training_epochs + 1):
            # metrics_map = evaluateEpoch(unwrap_DDP(self._model), self._validate_metric_fn_map, valid_dataloader,
            #                             self._tensorboard_writer,cur_epoch=epoch,
            #                             epoch_steps=epoch_steps,test_flag=False,device=self.device,rank=self._rank)
            if cfg.SOLVER.LR_DECAY_STAGES != "NONE":
                lr = adjust_learning_rate(epoch, self._optimizer, cfg)
            trainEpoch(self._model, self._validate_metric_fn_map, self._optimizer, train_dataloader,
                       self._tensorboard_writer,epoch,epoch_steps,self._logging_interval,self.device,self._rank)
            metrics_map = evaluateEpoch(unwrap_DDP(self._model), self._validate_metric_fn_map, valid_dataloader,
                                        self._tensorboard_writer,cur_epoch=epoch,
                                        epoch_steps=epoch_steps,test_flag=False,device=self.device,rank=self._rank)

            ###save checkpoint###
            self._model.train()

            state = {
                "epoch": epoch,
                "model": unwrap_DDP(self._model).state_dict(),
                # "optimizer": self._optimizer.state_dict(),
                "best_metric": self._best_metrics}
            if isinstance(unwrap_DDP(self._model), TransferrableModel):
                state["backbone"] = unwrap_DDP(self._model).backbone.state_dict()
            else:
                state["backbone"] = state["model"]
            torch.save(state, os.path.join(model_dir, "latest.pth"))
            torch.save(state["backbone"], os.path.join(model_dir, "backbone_latest.pth"))
            if epoch % cfg.EXPERIMENT.MODEL_SAVE_INTERVASL == 0:
                torch.save(state, os.path.join(model_dir, f"epoch_{epoch}.pth"))
                torch.save(state["backbone"], os.path.join(model_dir, f"backbone_epoch_{epoch}.pth"))
            if metrics_map[self._early_stop_metric] >= self._best_metrics:
                torch.save(state, os.path.join(model_dir,"best.pth"))
                torch.save(state["backbone"], os.path.join(model_dir,"backbone_best.pth"))
                self._best_metrics = metrics_map[self._early_stop_metric]

            #####################
            self._tensorboard_writer.flush()
            if self._early_stopping is not None:
                self._early_stopping(metrics_map[self._early_stop_metric], self._best_metrics)
                if self._early_stopping.early_stop:
                    logging.warning("Early stop after epoch:%s, the best acc is %s" % (epoch,
                                        self._early_stopping.optimal_metric))
                    break

class Evaluator:
    ''' The Evaluator

    '''
    def __init__(self,metric_fn_map,tensorboard_writer,device):
        ''' Init method

        :param metric_fn_map: metric function map, which map metric name to metric function
        :param tensorboard_writer: tensorboard writer
        '''
        self._metric_fn_map = metric_fn_map
        self._tensorboard_writer = tensorboard_writer
        self.device = device

    def evaluate(self,model, dataloader):
        ''' evaluate a model with test dataset

        :param model: the evaluated model
        :param dataloader: the dataloader of test dataset
        :return: metric_value_maps
        '''
        return evaluateEpoch(model, self._metric_fn_map, dataloader,self._tensorboard_writer,
                             cur_epoch=0,epoch_steps=0,test_flag=True,device=self.device,rank=-1)
    def __str__(self):
        _str = "Trainer: metric_fn_map:%s\n" % self._metric_fn_map
        _str += "\ttensorboard_writer:%s\n" % self._tensorboard_writer
        return _str
