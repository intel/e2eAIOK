#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/28/2022 8:39 AM

import logging

from attr import has
import torch
import datetime, time
import torch.nn
from torch.nn.parallel import DistributedDataParallel
import contextlib
import os
import numpy as np
from collections.abc import Iterable

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
    unwrapped_model = unwrap_DDP(model)
    with context:
        for (cur_step,(data, label)) in enumerate(train_dataloader):
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
            loss_value = unwrapped_model.loss(output, label)
            loss_value.backward()
            if cur_step % cfg.experiment.log_interval_step == 0:
                if is_transferrable: 
                    metric_values = unwrapped_model.get_training_metrics(output, label, loss_value, metric_fn_map)
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

def saveLogitsEpoch(model, dataloader, cur_epoch, epoch_steps, cfg,
                    device="cpu",rank=-1):
    ''' save teacher logits for distiller

    :param model: the pretrained teacher model
    :param dataloader: dataloader
    :param cur_epoch: current epoch
    :param epoch_steps: steps per step
    :param cfg: configurations
    :param device: running on cpu or gpu
    :param rank: rank for distributed training (-1 for non-distributed training)
    '''
    start_time = time.time()
    topk = cfg.distiller.logits_topk
    with torch.no_grad():
        model.eval()
        logits_manager = dataloader.dataset.get_manager()

        #################### iterate on dataset ##############
        for idx, ((data, label), (keys, seeds)) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            output = output.logits if cfg.model.type == "vit_base_224_in21k_ft_cifar100" else output
            
            if topk == 0: # save all logits
                values = output.detach().to(device='cpu', dtype=torch.float32)

                seeds = seeds.numpy()
                values = values.numpy() 
                assert seeds.dtype == np.int32, seeds.dtype
                assert values.dtype == np.float32, values.dtype

                for key, seed, value in zip(keys, seeds, values):
                    bstr = seed.tobytes() + value.tobytes()
                    logits_manager.write(key, bstr)
            elif topk > 0: # only save topk logits
                softmax_prob = torch.softmax(output, -1)
                values, indices = softmax_prob.topk(k=topk, dim=-1, largest=True, sorted=True)
                values = values.detach().to(device='cpu', dtype=torch.float16)
                indices = indices.detach().to(device='cpu', dtype=torch.int16)

                seeds = seeds.numpy()
                values = values.numpy()
                indices = indices.numpy()
                assert seeds.dtype == np.int32, seeds.dtype
                assert indices.dtype == np.int16, indices.dtype
                assert values.dtype == np.float16, values.dtype

                for key, seed, indice, value in zip(keys, seeds, indices, values):
                    bstr = seed.tobytes() + value.tobytes() + indice.tobytes()
                    logits_manager.write(key, bstr)
            if idx % cfg.experiment.log_interval_step == 0:
                dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
                print(f"{dt} Epoch:{cur_epoch}, save {idx}/{epoch_steps}")
                logging.info(f"{dt} Epoch:{cur_epoch}, save {idx}/{epoch_steps}")
    print(f"Epoch {cur_epoch} took {time.time()-start_time} seconds")
    logging.info(f"Epoch {cur_epoch} took {time.time()-start_time} seconds")

def checkLogitsEpoch(model, dataloader, cur_epoch, epoch_steps, num_classes,
                        tensorboard_writer, cfg, device="cpu",rank=-1):
    ''' check saved teacher logits for distiller

    :param model: the evaluated model
    :param dataloader: dataloader
    :param cur_epoch : current epoch
    :param epoch_steps: steps per step
    :param num_classes: number of prediction classes
    :param tensorboard_writer: tensorboard writer
    :param cfg: configurations
    :param device: running on cpu or gpu
    :param rank: rank for distributed training (-1 for non-distributed training)
    '''
    start_time = time.time()
    topk = cfg.distiller.logits_topk
    with torch.no_grad():
        model.eval()  # set evaluating flag
        logits_manager = dataloader.dataset.get_manager()

        loss_value = 0
        metric_values = {}
        #################### iterate on dataset ##############
        for idx, ((data, label), save_values) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            batch_size = data.size(0)
            output = output.logits if cfg.model.type == "vit_base_224_in21k_ft_cifar100" else output

            logits_value = output.detach().to(device='cpu', dtype=torch.float32)
            if topk == 0: 
                save_logits_value, _ = save_values
                save_logits_value = save_logits_value.float()
            elif topk > 0: 
                save_logits_value_topk, save_logits_index_topk, _ = save_values

                ### compare 1
                softmax_prob = torch.softmax(output, -1)
                logits_value_topk, logits_indices_topk = softmax_prob.topk(k=topk, dim=-1, largest=True, sorted=True)
                logits_value_topk = logits_value_topk.detach().to(device='cpu', dtype=torch.float16)
                logits_indices_topk = logits_indices_topk.detach().to(device='cpu', dtype=torch.int16)
                metric_values["indices_diff"] = torch.count_nonzero((logits_indices_topk != save_logits_index_topk)).item() / batch_size
                metric_values["value_topk_diff"] = (logits_value_topk - save_logits_value_topk).abs().sum().item() / batch_size

                ### compare 2
                save_logits_index_topk = save_logits_index_topk.long()
                save_logits_value_topk = save_logits_value_topk.float()
                minor_value = (1.0 - save_logits_value_topk.sum(-1, keepdim=True)
                            ) / (num_classes - topk)
                minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
                save_logits_value = minor_value.scatter_(-1, save_logits_index_topk, save_logits_value_topk)
            metric_values["value_diff"] = (logits_value - save_logits_value).abs().sum().item() / batch_size

            add_tensorboard_metric(tensorboard_writer,'Train',metric_values,cur_epoch,idx,epoch_steps=epoch_steps)
    print(f"Epoch {cur_epoch} took {time.time()-start_time} seconds")
    logging.info(f"Epoch {cur_epoch} took {time.time()-start_time} seconds")

def trainEpochwithLogits(model, metric_fn_map, optimizer, train_dataloader, num_classes,  
               tensorboard_writer,cur_epoch,epoch_steps,cfg,
               profiler=None, warmup_scheduler=None,device='cpu',rank=-1,is_transferrable=False):
    ''' train one epoch for distiller with loading teacher logits

    :param model: the training model
    :param metric_fn_map:  metric function map, which map metric name to metric function
    :param optimizer: the optimizer
    :param train_dataloader: train dataloader
    :param num_classes: number of prediction classes
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
    topk = cfg.distiller.logits_topk
    model.train()  # set training flag
    context = profiler if profiler is not None else contextlib.nullcontext()
    unwrapped_model = unwrap_DDP(model)
    with context:
        for cur_step, ((data, label), save_values) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)

            if topk == 0:
                logits_value, _ = save_values
                outputs_teacher = logits_value.float()
            elif topk > 0:
                logits_value, logits_index, _ = save_values
                logits_index = logits_index.long()
                logits_value = logits_value.float()
                minor_value = (1.0 - logits_value.sum(-1, keepdim=True)
                            ) / (num_classes - topk)
                minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
                outputs_teacher = minor_value.scatter_(-1, logits_index, logits_value)
            output.distiller_output = outputs_teacher

            loss_value = unwrapped_model.loss(output, label)
            loss_value.backward()

            if cur_step % cfg.experiment.log_interval_step == 0:
                metric_values = unwrapped_model.get_training_metrics(output, label, loss_value, metric_fn_map)
                add_tensorboard_metric(tensorboard_writer, 'Train', metric_values, cur_epoch,cur_step,epoch_steps,rank)
        
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
        unwrapped_model = unwrap_DDP(self._model)
        backbone = unwrapped_model.backbone if self._is_transferrable else unwrapped_model

        initial_epoch = self._cfg.solver.start_epoch
        if resume:
            state = torch.load(os.path.join(model_dir,"latest.pth"), map_location=self._device)
            initial_epoch = state["epoch"] + 1
            self._best_metrics_value = state["best_metric"]
            unwrapped_model.load_state_dict(state["model"])
            if self._rank < 0:
                self._optimizer.load_state_dict(state["optimizer"])
                self._scheduler.load_state_dict(state["scheduler"])
        if self._cfg.distiller.save_logits:
            initial_epoch = self._cfg.distiller.save_logits_start_epoch

        train_dataset = train_dataloader.dataset
        for epoch in range(initial_epoch, self._cfg.solver.epochs + 1):
            start_time = time.time()
            
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)

            ###### save teacher logits for distiller
            if self._cfg.distiller.save_logits: 
                saveLogitsEpoch(self._model, train_dataloader, epoch, epoch_steps, self._cfg,
                                 self._device,self._rank)
                continue

            ###### check teacher logits for distiller
            if self._cfg.distiller.check_logits: 
                checkLogitsEpoch(self._model, train_dataloader, epoch, epoch_steps, num_classes,
                                    self._tensorboard_writer, self._cfg, self._device, self._rank)
                continue

            if type(self._scheduler) is  torch.optim.lr_scheduler.ReduceLROnPlateau:
                last_lr = [item['lr'] for item in self._scheduler.optimizer.state_dict()['param_groups']]
            else:
                last_lr = self._scheduler.get_last_lr()

            print("Epoch [%s] lr: %s" % (epoch, last_lr))
            logging.info("Epoch [%s] lr: %s" % (epoch, last_lr))
            ###### train and evaluate for on epoch
            if self._cfg.distiller.use_saved_logits:
                trainEpochwithLogits(self._model, self._validate_metric_fn_map, self._optimizer, train_dataloader, num_classes,
                            self._tensorboard_writer, epoch, epoch_steps, self._cfg,
                            self._training_profiler, self._warmup_scheduler, self._device,self._rank, self._is_transferrable)
            else:
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
            ###### save checkpoint
            state = {
                "epoch": epoch,
                "best_metric": self._best_metrics_value,
                "model": unwrapped_model.state_dict(),
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
            if self._early_stopping is not None:
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
