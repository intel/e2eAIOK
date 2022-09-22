#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/28/2022 8:39 AM

import logging
import torch
import datetime
import torch.nn
from torch.nn.parallel import DistributedDataParallel
import contextlib


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
               tensorboard_writer,profiler,cur_epoch,epoch_steps,logging_interval,rank,is_transferrable):
    ''' train one epoch

    :param model: the training model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param optimizer: the optimizer
    :param train_dataloader: train dataloader
    :param epoch_steps: how many steps of an epoch
    :param tensorboard_writer: tensorboard writer
    :param profiler: profiler
    :param cur_epoch: current epoch
    :param logging_interval: logging interval durring training
    :param rank: rank for distributed training (-1 for non-distributed training)
    :param is_transferrable: is model transferrable
    :return:
    '''
    model.train()  # set training flag
    context = profiler if profiler is not None else contextlib.nullcontext()
    unwrapped_model = unwrap_DDP(model)
    with context:
        for (cur_step,(data, label)) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss_value = unwrapped_model.loss(output, label)
            loss_value.backward()
            if cur_step % logging_interval == 0:
                if is_transferrable: 
                    metric_values = unwrapped_model.get_training_metrics(output, label, loss_value, metric_fn_map)
                else:
                    metric_values = {"loss": loss_value}
                    for (metric_name, metric_fn) in sorted(metric_fn_map.items()):
                        metric_value = metric_fn(output, label)
                        metric_values[metric_name] = metric_value
                add_tensorboard_metric(tensorboard_writer, 'Train', metric_values, cur_epoch,cur_step,epoch_steps,rank)

            optimizer.step()
            if context is profiler:
                context.step()

def evaluateEpoch(model, metric_fn_map, dataloader,tensorboard_writer,profiler,
                  cur_epoch,epoch_steps,test_flag,rank,original_output_position):
    ''' evaluate epoch

    :param model: the evaluated model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param dataloader: dataloader
    :param tensorboard_writer : tensorboard writer
    :param profiler: profiler
    :param cur_epoch : current epoch
    :param epoch_steps: steps per step
    :param test_flag : whether is test or validation
    :param rank: rank for distributed training (-1 for non-distributed training)
    :param original_output_position: original output position in a tuple for TransferrableModel. If none, then output is the original one.
    :return: metric_value_maps
    '''
    datasetName = 'Test' if test_flag else 'Validation'
    context = profiler if profiler is not None else contextlib.nullcontext()
    with torch.no_grad():
        with context:
            model.eval()  # set evaluating flag
            loss_value = 0
            metric_values = {}
            sample_num = 0
            #################### iterate on dataset ##############
            for data, label in dataloader:
                # data, target = data.cuda(), target.cuda()
                output = model(data)
                if original_output_position is not None:
                    output = output[original_output_position]
                    
                batch_size = data.size(0)
                sample_num += batch_size
                loss_value += model.loss(output, label).item() * batch_size
                for (metric_name,metric_fn) in metric_fn_map.items():
                    metric_value = metric_fn(output,label)
                    if metric_name not in metric_values:
                        metric_values[metric_name] = metric_value * batch_size
                    else:
                        metric_values[metric_name] += metric_value * batch_size
                if context is profiler:
                    context.step()
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
    def __init__(self, model, optimizer, scheduler,early_stopping, validate_metric_fn_map,early_stop_metric,training_epochs,
                 tensorboard_writer,training_profiler,logging_interval,rank=-1,is_transferrable=False):
        ''' Init method

        :param model: the trained model
        :param optimizer: optimizer
        :param early_stopping: for early stopping
        :param scheduler: learning rate scheduler
        :param validate_metric_fn_map: metric function map, which map metric name to metric function
        :param early_stop_metric: metric name for early stopping
        :param training_epochs: max training epochs
        :param tensorboard_writer: tensorboard writer
        :param training_profiler : training profiler
        :param logging_interval: training logging interval
        :param rank: rank for distributed training (-1 for non-distributed training)
        :param is_transferrable: is transferrable
        '''
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._early_stopping = early_stopping
        self._validate_metric_fn_map = validate_metric_fn_map
        self._early_stop_metric = early_stop_metric
        self._training_epochs = training_epochs
        self._tensorboard_writer = tensorboard_writer
        self._training_profiler = training_profiler
        self._logging_interval = logging_interval
        self._rank = rank
        self._is_transferrable = is_transferrable

        if early_stop_metric not in validate_metric_fn_map:
            raise RuntimeError("early stop metric [%s] not in validate_metric_fn_map keys [%s]"%(
                early_stop_metric,",".join(validate_metric_fn_map.keys())
            ))

    def __str__(self):
        _str = "Trainer: model:%s\n"%self._model
        _str += "\toptimizer:%s\n"%self._optimizer
        _str += "\tscheduler:%s\n"%self._scheduler
        _str += "\tearly_stopping:%s\n" % self._early_stopping
        _str += "\tvalidate_metric_fn_map:%s\n" % self._validate_metric_fn_map
        _str += "\tearly_stop_metric:%s\n" % self._early_stop_metric
        _str += "\ttraining_epochs:%s\n" % self._training_epochs
        _str += "\ttensorboard_writer:%s\n" % self._tensorboard_writer
        _str += "\ttraining_profiler:%s\n"%self._training_profiler
        _str += "\tlogging_interval:%s\n" % self._logging_interval
        _str += "\trank:%s\n" % self._rank
        return _str

    def train(self, train_dataloader,epoch_steps,valid_dataloader,model_path):
        ''' train function, and save the best trained model to model_path

        :param train_dataloader: train dataloader
        :param epoch_steps: steps per epoch
        :param valid_dataloader: validation dataloader
        :param model_path: model path
        :return:
        '''
        if self._is_transferrable:
            backbone = unwrap_DDP(self._model).backbone
            original_output_position = 0
        else:
            backbone = unwrap_DDP(self._model)
            original_output_position = None

        for epoch in range(1, self._training_epochs + 1):
            trainEpoch(self._model, self._validate_metric_fn_map, self._optimizer, train_dataloader,
                       self._tensorboard_writer, self._training_profiler if epoch == 1 else None,
                       epoch,epoch_steps,self._logging_interval,self._rank,self._is_transferrable)

            metrics_map = evaluateEpoch(backbone, self._validate_metric_fn_map, valid_dataloader,
                                        self._tensorboard_writer,None,cur_epoch=epoch,
                                        epoch_steps=epoch_steps,test_flag=False,rank=self._rank,
                                        original_output_position=original_output_position)
            self._tensorboard_writer.flush()
            self._scheduler.step()
            print("epoch [%s] lr: %s" % (epoch, self._scheduler.get_last_lr()))
            if self._early_stopping is not None:
                self._early_stopping(metrics_map[self._early_stop_metric], backbone.state_dict())
                if self._early_stopping.early_stop:
                    logging.warning("Early stop after epoch:%s, the best acc is %s" % (epoch,
                                           self._early_stopping.optimal_metric))
                    break
        if self._early_stopping is not None and self._early_stopping.optimal_model is not None:
            torch.save(self._early_stopping.optimal_model, model_path)
        else:
            torch.save(backbone.state_dict(), model_path)

class Evaluator:
    ''' The Evaluator

    '''
    def __init__(self,metric_fn_map,tensorboard_writer,profiler):
        ''' Init method

        :param metric_fn_map: metric function map, which map metric name to metric function
        :param tensorboard_writer: tensorboard writer
        :param profiler: profiler
        '''
        self._metric_fn_map = metric_fn_map
        self._tensorboard_writer = tensorboard_writer
        self._profiler = profiler

    def evaluate(self,model, dataloader):
        ''' evaluate a model with test dataset

        :param model: the evaluated model
        :param dataloader: the dataloader of test dataset
        :return: metric_value_maps
        '''
        return evaluateEpoch(model, self._metric_fn_map, dataloader,self._tensorboard_writer,self._profiler,
                             cur_epoch=0,epoch_steps=0,test_flag=True,rank=-1,original_output_position=None)
    def __str__(self):
        _str = "Trainer: metric_fn_map:%s\n" % self._metric_fn_map
        _str += "\ttensorboard_writer:%s\n" % self._tensorboard_writer
        _str += "\tprofiler:%s\n" % self._profiler
        return _str
