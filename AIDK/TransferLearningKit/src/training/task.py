#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/19/2022 10:08 AM
import torch
import torchvision.models
import datetime
from torch.utils.tensorboard import SummaryWriter
from engine_core.backbone.factory import createBackbone
from engine_core.adapter.factory import createAdapter
from engine_core.distiller import KD, DKD
from engine_core.distiller.utils import logits_wrap_dataset
from engine_core.finetunner.basic_finetunner import BasicFinetunner
from engine_core.transferrable_model import *
from .train import Trainer,Evaluator
from dataset import get_dataset, get_dataloader
import torch.optim as optim
from .utils import EarlyStopping,Timer, WarmUpLR
import logging
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer as ZeRO
from torch.distributed.algorithms.join import Join
from torch.profiler import profile, ProfilerActivity
import intel_extension_for_pytorch as ipex
import os

def trace_handler(trace_file,p):
    ''' profile trace handler

    :param trace_file: output trace file
    :param p: profiler
    :return:
    '''
    logging.info("trace_output:%s" % p.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    p.export_chrome_trace(trace_file + str(p.step_num) + ".json")

class Task:
    ''' A whole training Task

    '''
    def __init__(self, cfg, model_save_path, loss, is_distributed):
        self._cfg = cfg
        self._loss = loss
        self._is_distributed = is_distributed
        self._model_dir = model_save_path
    def _create_dataloader(self):
        ''' create dataloader

        :return: (train_loader,validate_loader,test_loader, num_classes, num_data)
        '''
        train_dataset, valid_dataset, test_dataset, num_classes, num_data = get_dataset(self._cfg)
        if self._cfg.distiller.save_logits or self._cfg.distiller.use_saved_logits or self._cfg.distiller.check_logits:
            train_dataset = logits_wrap_dataset(train_dataset, logits_path=self._cfg.distiller.logits_path, num_classes=num_classes, \
                                            save_logits=self._cfg.distiller.save_logits, topk=self._cfg.distiller.logits_topk)
        train_loader, validate_loader, test_loader = get_dataloader(self._cfg, \
                            train_dataset, valid_dataset, test_dataset, self._is_distributed, self._cfg.optimize.enable_ipex)
        self._train_loader = train_loader     # may be use by other component
        self._epoch_steps = len(train_loader) # may be use by other component
        self._num_classes = num_classes 
        logging.info("epoch_steps:%s" % self._epoch_steps)
        return train_loader, validate_loader, test_loader, num_classes, num_data
    def _create_backbone(self):
        ''' create backbone model

        :return: a backbone model
        '''    
        backbone = createBackbone(self._cfg.model.type, num_classes = self._num_classes, pretrain = self._cfg.model.pretrain)
        set_attribute("model", backbone, "loss", self._loss)
        num_params = sum(param.numel() for param in backbone.parameters())
        self._backbone = backbone # may be use by other component
        logging.info('backbone:%s' % backbone)
        logging.info("Model params: %s" % num_params)
        print("Model params: ", num_params)
        return backbone
    def _create_finetuner(self):
        ''' create finetuner

        :return a finetuner
        '''
        if self._cfg.finetuner.type == "":
            finetuner = None
        elif self._cfg.finetuner.type == "Basic":
            pretrained_model = createBackbone(self._cfg.model.type, num_classes=self._cfg.finetuner.pretrained_num_classes, pretrain=self._cfg.finetuner.pretrain)
            finetuner = BasicFinetunner(pretrained_model, is_frozen=self._cfg.finetuner.frozen)
        else:
            logging.error("[%s] is not supported"%self._cfg.finetuner.type)
            raise NotImplementedError("[%s] is not supported"%self._cfg.finetuner.type)    
        self._finetuner = finetuner
        logging.info('finetuner:%s' % finetuner)
        return finetuner
    def _create_distiller(self):
        ''' create distiller
        
        :return a distiller
        '''
        if self._cfg.distiller.type == "":
            distiller = None
        else:
            teacher_model = createBackbone(self._cfg.distiller.teacher.type, num_classes = self._num_classes, pretrain = self._cfg.distiller.teacher.pretrain)
            if self._cfg.distiller.teacher.type != "vit_base_224_in21k_ft_cifar100":
                teacher_model = extract_distiller_adapter_features(teacher_model,self._cfg.distiller.feature_layer_name,self._cfg.adapter.feature_layer_name)
            if self._cfg.distiller.type == "kd":
                distiller = KD(pretrained_model = teacher_model, 
                                temperature = self._cfg.kd.temperature,
                                is_frozen = self._cfg.distiller.teacher.frozen, 
                                use_saved_logits = self._cfg.distiller.use_saved_logits,
                                topk=self._cfg.distiller.logits_topk, 
                                num_classes=self._num_classes, 
                                teacher_type = self._cfg.distiller.teacher.type)
            elif self._cfg.distiller.type == "DKD":
                distiller = DKD(pretrained_model = teacher_model, 
                                alpha = self._cfg.dkd.alpha, beta = self._cfg.dkd.beta,
                                temperature = self._cfg.kd.temperature, warmup = self._cfg.dkd.warmup,
                                is_frozen = self._cfg.distiller.teacher.frozen, 
                                use_saved_logits = self._cfg.distiller.use_saved_logits,
                                topk=self._cfg.distiller.logits_topk, 
                                num_classes=self._num_classes, 
                                teacher_type = self._cfg.distiller.teacher.type)
            else:
                logging.error("[%s] is not supported"%self._cfg.distiller.type)
                raise NotImplementedError("[%s] is not supported"%self._cfg.distiller.type)
        self._distiller = distiller
        logging.info('distiller:%s' % distiller)
        return distiller
    def _create_adapter(self):
        ''' create adapter

        :return an adapter
        '''
        if self._cfg.adapter.type == "":
            adapter = None
        elif self._cfg.adapter.type == "CDAN":
            adapter = createAdapter('CDAN', input_size=self._cfg.adapter.feature_size * self._num_classes, hidden_size=self._cfg.adapter.feature_size,
                                    dropout=0.0, grl_coeff_alpha=5.0, grl_coeff_high=1.0, max_iter=self._epoch_steps,
                                    backbone_output_size=self._num_classes, enable_random_layer=0, enable_entropy_weight=0)
        else:
            logging.error("[%s] is not supported"%self._cfg.adapter.type)
            raise NotImplementedError("[%s] is not supported"%self._cfg.adapter.type)
        self._adapter = adapter
        logging.info('adapter:%s' % adapter)
        return adapter
    def _create_transfer_model(self, backbone, finetuner, distiller, adapter):
        ''' create transferrable model

        :return: a transferrable model
        '''
        if self._cfg.experiment.strategy == "":
            strategy = None
            model = backbone
        elif self._cfg.experiment.strategy == "OnlyFinetuneStrategy":
            strategy = TransferStrategy.OnlyFinetuneStrategy
            model = make_transferrable_with_finetune(backbone, self._loss, finetunner=finetuner)
        elif self._cfg.experiment.strategy == "OnlyDistillationStrategy":
            strategy = TransferStrategy.OnlyDistillationStrategy
            model = make_transferrable_with_knowledge_distillation(backbone,self._loss,distiller,
                                                   self._cfg.distiller.feature_size,self._cfg.distiller.feature_layer_name,
                                                   True,self._cfg.experiment.loss.backbone,self._cfg.experiment.loss.distiller)
        elif self._cfg.experiment.strategy == "OnlyDomainAdaptionStrategy":
            strategy = TransferStrategy.OnlyDomainAdaptionStrategy
        elif self._cfg.experiment.strategy == "FinetuneAndDomainAdaptionStrategy":
            strategy = TransferStrategy.FinetuneAndDomainAdaptionStrategy
        elif self._cfg.experiment.strategy == "DistillationAndDomainAdaptionStrategy":
            strategy = TransferStrategy.DistillationAndDomainAdaptionStrategy
        else:
            raise NotImplementedError("[%s] is not supported"%self._cfg.experiment.strategy)
        
        if self._is_distributed:
            logging.info("training with DistributedDataParallel")
            model = DDP(model)
        logging.info("model:%s"%model)
        self._model = model # may be use by other component
        return model, strategy
    def _create_optimizer(self):
        ''' create optimizer

        :return: an optimizer
        '''
        if self._cfg.experiment.strategy == "OnlyFinetuneStrategy":
            finetuned_state_keys = ["backbone.%s"%name for name in self._finetuner.finetuned_state_keys] # add component prefix
            finetuner_params = {'params':[p for (name, p) in self._model.named_parameters() if p.requires_grad and name in finetuned_state_keys],
                                'lr': self._cfg.finetuner.finetuned_lr}
            remain_params = {'params':[p for (name, p) in self._model.named_parameters() if p.requires_grad and name not in finetuned_state_keys],
                                'lr': self._cfg.solver.optimizer.lr}
            logging.info("[%s] params set finetuner finetuned learning rate[%s]" % (len(finetuner_params['params']), self._cfg.finetuner.finetuned_lr))
            logging.info("[%s] params set common learning rate [%s]" % (len(remain_params['params']), self._cfg.solver.optimizer.lr))
            assert len(finetuner_params) > 0,"Empty finetuner_params"
            parameters = [finetuner_params,remain_params]
        else:
            remain_params = {'params': [p for p in self._model.parameters() if p.requires_grad],
                             'lr': self._cfg.solver.optimizer.lr}
            logging.info("[%s] params set common learning rate [%s]" % (len(remain_params), self._cfg.solver.optimizer.lr))
            parameters = [remain_params]
        if self._is_distributed:
            logging.info("training with DistributedDataParallel")
            if self._cfg.solver.optimizer.type == "SGD":
                optimizer = ZeRO(parameters, optim.SGD, weight_decay=self._cfg.solver.optimizer.weight_decay,momentum=self._cfg.solver.optimizer.momentum)
            else:
                logging.error("[%s] is not supported"%self._cfg.solver.optimizer.type)
                raise NotImplementedError("[%s] is not supported"%self._cfg.solver.optimizer.type)
        else:
            if self._cfg.solver.optimizer.type == "SGD":
                optimizer = optim.SGD(parameters,lr=self._cfg.solver.optimizer.lr, weight_decay=self._cfg.solver.optimizer.weight_decay,momentum=self._cfg.solver.optimizer.momentum)
            else:
                logging.error("[%s] is not supported"%self._cfg.solver.optimizer.type)
                raise NotImplementedError("[%s] is not supported"%self._cfg.solver.optimizer.type)
        logging.info("optimizer:%s"%optimizer)
        self._optimizer = optimizer # may be use by other component
        return optimizer
    def _create_lr_scheduler(self):
        ''' create lr_scheduler

        :return: a lr_scheduler
        '''
        if self._cfg.solver.scheduler.type == "":
            scheduler = None
        elif self._cfg.solver.scheduler.type == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=self._cfg.solver.scheduler.lr_decay_rate)
        elif self._cfg.solver.scheduler.type == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=self._cfg.solver.scheduler.MultiStepLR.lr_decay_stages, gamma=self._cfg.solver.scheduler.lr_decay_rate)
        elif self._cfg.solver.scheduler.type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._cfg.solver.scheduler.CosineAnnealingLR.T_max)
        elif self._cfg.solver.scheduler.type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode="max",factor=self._cfg.solver.scheduler.lr_decay_rate, patience=self._cfg.solver.scheduler.ReduceLROnPlateau.patience, \
                                                            verbose=True, threshold_mode='abs')
        else:
            logging.error("[%s] is not supported"%self._cfg.solver.scheduler.type)
            raise NotImplementedError("[%s] is not supported"%self._cfg.solver.scheduler.type)
        self._scheduler = scheduler
        return scheduler
    def _create_warmup_lr_scheduler(self):
        ''' create warm up lr_scheduler

        :return: a warm up  lr_scheduler
        '''
            
        warmup_scheduler = None
        if self._cfg.solver.warmup > 0:
            warmup_scheduler = WarmUpLR(self._optimizer, self._epoch_steps * self._cfg.solver.warmup)
        
        return warmup_scheduler
    def _create_tensorboard_writer(self):
        ''' create tensorboard_writer

        :return: a tensorboard_writer
        '''
        tensorboard_writer = SummaryWriter(self._cfg.experiment.tensorboard_dir)
        logging.info('tensorboard_writer :%s' % tensorboard_writer)
        return tensorboard_writer
    def _create_early_stopping(self):
        ''' create early_stopping

        :return: an early_stopping
        '''
        if self._cfg.solver.early_stop.flag:
            early_stopping = EarlyStopping(tolerance_epoch=self._cfg.solver.early_stop.tolerance_epoch, delta=self._cfg.solver.early_stop.delta, 
                                        is_max=self._cfg.solver.early_stop.is_max, limitation=self._cfg.solver.early_stop.limitation)
            logging.info('early_stopping :%s' % early_stopping)
        else:
            early_stopping = None
        return early_stopping
    def _create_profiler(self):
        ''' create a profiler

        :return: (training_profiler,inference_profiler)
        '''
        def parse_activities(activity_str):
            result = set()
            for item in activity_str.lower().split(","):
                if item == 'cpu':
                    result.add(ProfilerActivity.CPU)
                elif item == 'gpu' or item == 'cuda':
                    result.add(ProfilerActivity.CUDA)
            return result

        activities = parse_activities(self._cfg.profiler.activities)
        schedule = torch.profiler.schedule(skip_first=self._cfg.profiler.skip_first,wait=self._cfg.profiler.wait,
            warmup=self._cfg.profiler.warmup,active=self._cfg.profiler.active,repeat=self._cfg.profiler.repeat)
        training_profiler = profile(activities=activities,schedule=schedule,
                           on_trace_ready=partial(trace_handler,self._cfg.profiler.trace_file_training))
        inference_profiler = profile(activities=activities, schedule=schedule,
                                   on_trace_ready=partial(trace_handler, self._cfg.profiler.trace_file_inference))
        logging.info("training_profiler:%s"%training_profiler)
        logging.info("inference_profiler:%s" % inference_profiler)
        return (training_profiler,inference_profiler)
    def _load_trained_model(self):
        ''' load the trained model

        :return: a trained model
        '''
        trained_model = createBackbone(self._cfg.model.type, num_classes=self._num_classes)
        model_saved_path = os.path.join(self._model_dir, "backbone_best.pth")
        if os.path.exists(model_saved_path):
            logging.info("load the best trained model")
            trained_model.load_state_dict(torch.load(model_saved_path,map_location="cpu"),strict=True)
        else:
            raise RuntimeError("Can not find [%s]"%(model_saved_path))

        set_attribute("trained_model", trained_model, "loss",self._loss)
        return trained_model
    def run(self, validate_metric_fn_map, rank, eval=False, resume=False):
        ''' Run task
        :param validate_metric_fn_map: metric for validation
        :param rank: rank
        :param eval: whether only for evaluate
        :param resume: whether resume for train
        Returns: validation metric. If has Earlystopping, using the best metric; Else, using the last metric.
        '''
        ######################### create components #########################
        (train_loader, validate_loader, test_loader, num_classes, num_data) = self._create_dataloader()
        distiller = self._create_distiller()
        if distiller is not None and (self._cfg.distiller.save_logits or self._cfg.distiller.check_logits):
            distiller.prepare_logits(train_loader, self._cfg.solver.epochs,
                                        start_epoch=self._cfg.distiller.save_logits_start_epoch,
                                        device = self._cfg.profiler.activities,
                                        save_flag=self._cfg.distiller.save_logits,
                                        check_flag=self._cfg.distiller.check_logits)
            return
        backbone = self._create_backbone()
        finetuner = self._create_finetuner()
        adapter = self._create_adapter()
        model,strategy = self._create_transfer_model(backbone, finetuner, distiller, adapter)
        is_transferrable = strategy != None 
        optimizer = self._create_optimizer()
        if self._cfg.optimize.enable_ipex:
            model = model.to(memory_format = torch.channels_last)
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            if is_transferrable:
                set_attribute("backbone",model.backbone, "loss", self._loss)
        
        lr_scheduler = self._create_lr_scheduler()
        warmup_scheduler = self._create_warmup_lr_scheduler()
        tensorboard_writer = self._create_tensorboard_writer()
        early_stopping = self._create_early_stopping()
        training_profiler, inference_profiler = self._create_profiler()
        ######################### Evaluate #########################
        evaluator = Evaluator(validate_metric_fn_map, tensorboard_writer, self._cfg, inference_profiler, self._cfg.profiler.activities)
        logging.info("evaluator:%s" % evaluator)
        if eval:
            with Timer():
                start = datetime.datetime.now()
                metric_values = evaluator.evaluate(model, test_loader)
                total_seconds = datetime.datetime.now() - start
                test_num = num_data["test"]
                print(f"test data: {test_num}, throughput: {test_num/total_seconds} samples/second")
                print(metric_values)
            return metric_values
        #################################### train and evaluate ###################
        trainer = Trainer(model, optimizer, lr_scheduler, validate_metric_fn_map, self._cfg.solver.early_stop.metric,
                            tensorboard_writer, self._cfg, 
                            warmup_scheduler=warmup_scheduler, early_stopping=early_stopping, is_transferrable=is_transferrable,
                            training_profiler=training_profiler, device=self._cfg.profiler.activities, rank=rank)      
        logging.info("trainer:%s" % trainer)
        with Timer():
            if self._is_distributed:
                with Join([model, optimizer]):
                    val_metric = trainer.train(train_loader, validate_loader, self._epoch_steps, self._model_dir, self._num_classes, resume)
            else:
                val_metric = trainer.train(train_loader, validate_loader, self._epoch_steps, self._model_dir, self._num_classes, resume)
        ################################### test ###################################
        if test_loader is not None:
            if (not (self._cfg.distiller.save_logits or self._cfg.distiller.check_logits)) and ((not self._is_distributed) or (self._is_distributed and rank == 0)): # only test once
                trained_model = self._load_trained_model()
                with Timer():
                    evaluator.evaluate(trained_model, test_loader)
        return val_metric
