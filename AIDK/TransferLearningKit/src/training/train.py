#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/28/2022 8:39 AM

import logging
import torch
import datetime, time
import torch.nn
import contextlib
import os
import numpy as np
from collections.abc import Iterable

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
               tensorboard_writer,cur_epoch,epoch_steps,cfg,
               profiler=None, warmup_scheduler=None,device='cpu',rank=-1,is_transferrable=False):
    ''' train one epoch

    :param model: the training model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param optimizer: the optimizer
    :param train_dataloader: train dataloader
    :param tensorboard_writer: tensorboard writer
    :param cur_epoch: current epoch
    :param epoch_steps: how many steps of an epoch
    :param cfg: configurations
    :param profiler: profiler
    :param warmup_scheduler: the scheduler for warmup
    :param device: running on cpu or gpu
    :param rank: rank for distributed training (-1 for non-distributed training)
    :param is_transferrable: is model transferrable
    :return:
    '''
    model.train()  # set training flag
    context = profiler if profiler is not None else contextlib.nullcontext()
    with context:
        for (cur_step,(data, label)) in enumerate(train_dataloader):
            '''
            Four cases of data
            Case 1 - basic: input
            Case 2 - distiller with logits: (input, logits)
            Case 3 - adapter: (input1, input2)  - to be supported
            Case 4 - distiller with logits and adapter: ((input1, logits1),(input2, logits2)) - to be supported
            '''
            if isinstance(data, torch.Tensor):
                data = data.to(device)
                label = label.to(device)
            elif isinstance(data, Iterable):
                assert len(data) == len(label), "data len[%s] must equal label len[%s]"%(len(data),len(label))
                for i in range(0,len(data)):
                    data[i] = data[i].to(device)
                    label[i] = label[i].to(device)
            else:
                raise RuntimeError("Known data type:%s"%type(data))
            optimizer.zero_grad()
            output = model(data)
            loss_value = model.loss(output, label)
            loss_value.backward()
            if cur_step % cfg.experiment.log_interval_step == 0:
                if is_transferrable: 
                    metric_values = model.get_training_metrics(output, label, loss_value, metric_fn_map)
                else:
                    metric_values = {"loss": loss_value}
                    for (metric_name, metric_fn) in sorted(metric_fn_map.items()):
                        metric_value = metric_fn(output, label)
                        metric_values[metric_name] = metric_value
                add_tensorboard_metric(tensorboard_writer, 'Train', metric_values, cur_epoch, cur_step, epoch_steps, rank)

            if cur_step in [0, epoch_steps - 1] or  cur_step % (cfg.experiment.log_interval_step * 10) == 0: # first iter, last iter and several middle iter.
                for (name, parameter) in model.named_parameters():
                    tensorboard_writer.add_histogram(name, parameter, cur_epoch * epoch_steps + cur_step)
                    if parameter.requires_grad:
                        tensorboard_writer.add_histogram("%s_Grad"%name, parameter.grad, cur_epoch * epoch_steps + cur_step)

            optimizer.step()
            if cur_epoch <= cfg.solver.warmup:
                warmup_scheduler.step()
            if context is profiler:
                context.step()

def evaluateEpoch(model, metric_fn_map, dataloader, 
                    tensorboard_writer, cur_epoch, cfg,
                    test_flag=True, profiler=None, device='cpu',rank=-1):
    ''' evaluate epoch

    :param model: the evaluated model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param dataloader: dataloader
    :param tensorboard_writer : tensorboard writer
    :param cur_epoch : current epoch
    :param cfg: configurations
    :param test_flag : whether is test or validation
    :param profiler: profiler
    :param device: running on cpu or gpu
    :param rank: rank for distributed training (-1 for non-distributed training)
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
            epoch_steps = len(dataloader)
            for (cur_step,(data, label)) in enumerate(dataloader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                output = output.logits if cfg.model.type == "vit_base_224_in21k_ft_cifar100" else output
                if isinstance(output, Iterable):
                    if not isinstance(output, torch.Tensor): # Tensor is Iterable
                        output = output[0]
                else:
                    raise RuntimeError("Known data type:%s"%type(data))

                batch_size = data.size(0)
                sample_num += batch_size
                loss_value += model.loss(output, label).item() * batch_size
                for (metric_name,metric_fn) in metric_fn_map.items():
                    metric_value = metric_fn(output,label)
                    if metric_name not in metric_values:
                        metric_values[metric_name] = metric_value * batch_size
                    else:
                        metric_values[metric_name] += metric_value * batch_size
                if cur_step % cfg.experiment.log_interval_step == 0:
                    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
                    print(f"{dt} {cur_step}/{epoch_steps}")
                    logging.info(f"{dt} {cur_step}/{epoch_steps}")
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
    def __init__(self, model, optimizer, scheduler, validate_metric_fn_map, best_metric,
                    tensorboard_writer, cfg,
                    warmup_scheduler=None, early_stopping=None, is_transferrable=False,
                    training_profiler=None, device='cpu',rank=-1):
        ''' Init method

        :param model: the trained model
        :param optimizer: optimizer
        :param scheduler: learning rate scheduler
        :param validate_metric_fn_map: metric function map, which map metric name to metric function
        :param best_metric: metric name for update best model
        :param tensorboard_writer: tensorboard writer
        :param cfg: configurations
        :param warmup_scheduler: the scheduler for warmup
        :param early_stopping: for early stopping
        :param is_transferrable: is transferrable
        :param training_profiler : training profiler
        :param device: running on cpu or gpu
        :param rank: rank for distributed training (-1 for non-distributed training)
        '''
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._validate_metric_fn_map = validate_metric_fn_map
        self._best_metric = best_metric
        self._tensorboard_writer = tensorboard_writer
        self._cfg = cfg
        self._warmup_scheduler = warmup_scheduler
        self._early_stopping = early_stopping
        self._is_transferrable = is_transferrable
        self._training_profiler = training_profiler
        self._device = device
        self._rank = rank

        self._best_metrics_value = 0.0
        
        if best_metric not in validate_metric_fn_map:
            raise RuntimeError("early stop metric [%s] not in validate_metric_fn_map keys [%s]"%(
                best_metric,",".join(validate_metric_fn_map.keys())
            ))

    def __str__(self):
        _str = "Trainer: model:%s\n"%self._model
        _str += "\toptimizer:%s\n"%self._optimizer
        _str += "\tscheduler:%s\n" % self._scheduler
        _str += "\tvalidate_metric_fn_map:%s\n" % self._validate_metric_fn_map
        _str += "\t_best_metric:%s\n" % self._best_metric
        _str += "\ttensorboard_writer:%s\n" % self._tensorboard_writer
        _str += "\tconfigurations:%s\n" % self._cfg
        _str += "\twarmup_scheduler:%s\n" % self._warmup_scheduler
        _str += "\tearly_stopping:%s\n" % self._early_stopping
        _str += "\tis_transferrable:%s\n" % self._is_transferrable
        _str += "\ttraining_profiler:%s\n" % self._training_profiler
        _str += "\tdevice:%s\n" % self._device
        _str += "\trank:%s\n" % self._rank
        _str += "\tbest_metrics_value:%s\n" % self._best_metrics_value
        _str += "\ttraining_epochs:%s\n" % self._cfg.solver.epochs
        _str += "\tlogging_interval:%s\n" % self._cfg.experiment.log_interval_step
        return _str

    def train(self, train_dataloader, valid_dataloader, epoch_steps, model_dir, num_classes=10, resume=False):
        ''' train function, and save the best trained model to self._cfg.experiment.model_save
        :param train_dataloader: train dataloader
        :param valid_dataloader: validation dataloader
        :param epoch_steps: steps per epoch
        :param model_dir: model saved dir
        :param num_classes: number of prediction classes
        :param resume: flag for whether resume pretrained model
        :return: validation metric. If has Earlystopping, using the best metric; Else, using the last metric.
        :return: validation metric. If has Earlystopping, using the best metric; Else, using the last metric.
        '''
        backbone = self._model.get_backbone() if self._is_transferrable else self._model

        initial_epoch = self._cfg.solver.start_epoch
        if resume:
            state = torch.load(os.path.join(model_dir,"latest.pth"), map_location=self._device)
            initial_epoch = state["epoch"] + 1
            self._best_metrics_value = state["best_metric"]
            self._model.load_state_dict(state["model"])
            if self._rank < 0:
                self._optimizer.load_state_dict(state["optimizer"])
                self._scheduler.load_state_dict(state["scheduler"])
        train_dataset = train_dataloader.dataset
        for epoch in range(initial_epoch, self._cfg.solver.epochs + 1):
            start_time = time.time()
            
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)

            ###### train and evaluate for on epoch
            trainEpoch(self._model, self._validate_metric_fn_map, self._optimizer, train_dataloader,
                            self._tensorboard_writer, epoch, epoch_steps, self._cfg,
                            self._training_profiler, self._warmup_scheduler, self._device,self._rank, self._is_transferrable)
            metrics_map = evaluateEpoch(backbone, self._validate_metric_fn_map, valid_dataloader,
                                        self._tensorboard_writer,cur_epoch=epoch, cfg=self._cfg,
                                        test_flag=False, profiler=self._training_profiler, device=self._device,rank=self._rank)
            
            ###### flush tensorboard and update scheduler
            self._tensorboard_writer.flush()
            if epoch > self._cfg.solver.warmup:
                if self._cfg.solver.scheduler.type == "ReduceLROnPlateau":
                    self._scheduler.step(metrics_map[self._best_metric])
                else:
                    self._scheduler.step()
           
            if self._rank <= 0: # non distributed training, or rank 0  in distributed training
                ###### save checkpoint
                state = {
                    "epoch": epoch,
                    "best_metric": self._best_metrics_value,
                    "model": self._model.state_dict(),
                    "optimizer": self._optimizer.state_dict() if self._rank < 0 else None,
                    "scheduler": self._scheduler.state_dict() if self._rank < 0 else None}
                torch.save(state, os.path.join(model_dir, "latest.pth"))
                torch.save(backbone.state_dict(), os.path.join(model_dir, "backbone_latest.pth"))
                if epoch % self._cfg.experiment.model_save_interval == 0:
                    torch.save(state, os.path.join(model_dir, f"epoch_{epoch}.pth"))
                    torch.save(backbone.state_dict(), os.path.join(model_dir, f"backbone_epoch_{epoch}.pth"))
                if metrics_map[self._best_metric] >= self._best_metrics_value:
                    torch.save(state, os.path.join(model_dir,"best.pth"))
                    torch.save(backbone.state_dict(), os.path.join(model_dir,"backbone_best.pth"))
                    self._best_metrics_value = metrics_map[self._best_metric]
                    print(f"Best Epoch: {epoch}")
                    logging.info(f"Best Epoch: {epoch}")

            print(f"Epoch {epoch} took {time.time()-start_time} seconds")
            logging.info(f"Epoch {epoch} took {time.time()-start_time} seconds")
            
            ###### check early stop
            if self._rank <= 0 and self._early_stopping is not None: # non distributed training, or rank 0  in distributed training
                self._early_stopping(metrics_map[self._best_metric], self._best_metrics_value)
                if self._early_stopping.early_stop:
                    logging.warning("Early stop after epoch:%s, the best acc is %s" % (epoch,self._best_metrics_value))
                    break
        
        return self._best_metrics_value    

class Evaluator:
    ''' The Evaluator

    '''
    def __init__(self,metric_fn_map, tensorboard_writer, cfg, profiler=None, device='cpu'):
        ''' Init method

        :param metric_fn_map: metric function map, which map metric name to metric function
        :param tensorboard_writer: tensorboard writer
        :param cfg: configurations
        :param profiler: profiler
        :param device: running on cpu or gpu
        '''
        self._metric_fn_map = metric_fn_map
        self._tensorboard_writer = tensorboard_writer
        self._cfg = cfg
        self._profiler = profiler
        self._device = device

    def evaluate(self,model, dataloader):
        ''' evaluate a model with test dataset

        :param model: the evaluated model
        :param dataloader: the dataloader of test dataset
        :return: metric_value_maps
        '''
        return evaluateEpoch(model, self._metric_fn_map, dataloader,
                            self._tensorboard_writer, cur_epoch=0, cfg=self._cfg,
                            test_flag=True, profiler=self._profiler, device=self._device,rank=-1)
    def __str__(self):
        _str = "Trainer: metric_fn_map:%s\n" % self._metric_fn_map
        _str += "\ttensorboard_writer:%s\n" % self._tensorboard_writer
        _str += "\tprofiler:%s\n" % self._profiler
        return _str
