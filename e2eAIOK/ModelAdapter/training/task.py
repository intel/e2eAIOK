import torch
import time
import logging
from types import MethodType
import os
import e2eAIOK.common.trainer.utils.utils as utils
from e2eAIOK.ModelAdapter.engine_core.adapter import createAdapter
from e2eAIOK.ModelAdapter.engine_core.distiller import KD, DKD
from e2eAIOK.ModelAdapter.engine_core.distiller.utils import logits_wrap_dataset
from e2eAIOK.ModelAdapter.engine_core.finetunner import BasicFinetunner
from e2eAIOK.ModelAdapter.engine_core.transferrable_model import *
from e2eAIOK.ModelAdapter.training import TorchTrainerMA
from e2eAIOK.ModelAdapter.dataset import createDataBuilder
from e2eAIOK.ModelAdapter.backbone import createBackbone
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist

class ModelAdapterTask:
    ''' A Model Adapter Task

    '''
    def __init__(self, cfg, model_save_path, is_distributed):
        self.cfg = cfg        
        self.model_dir = model_save_path
        self.is_distributed = is_distributed
        self.create_loss()
        self.get_device()
    
    def get_device(self):
        self.device = utils.get_device(self.cfg)

    def create_loss(self):
        self.loss = utils.create_criterion(self.cfg)
    
    def create_metric(self):
        metric = {self.cfg.eval_metric:utils.create_metric(self.cfg)}
        return metric

    def create_dataloader(self):
        ''' create dataloader

        :return: (train_loader,validate_loader,test_loader)
        '''
        dataBuilder, num_classes = createDataBuilder(self.cfg)
        dataBuilder.prepare_dataset()
        if "distill" in self.cfg.experiment.strategy.lower() and (self.cfg.distiller.save_logits or \
            self.cfg.distiller.use_saved_logits or self.cfg.distiller.check_logits):
            dataBuilder.dataset_train = logits_wrap_dataset(dataBuilder.dataset_train, logits_path=self.cfg.distiller.logits_path, \
                            num_classes=num_classes, save_logits=self.cfg.distiller.save_logits, topk=self.cfg.distiller.logits_topk)
        dataloaders = dataBuilder.get_dataloader()
        if len(dataloaders) == 2:
            train_loader, validate_loader = dataloaders
            test_loader = validate_loader
        elif len(dataloaders) == 3:
            train_loader, validate_loader, test_loader = dataloaders
        self.epoch_steps = len(train_loader) # may be use by other component
        self.num_classes = num_classes 
        logging.info("data builder:" + str(dataBuilder))
        logging.info("num_classes:" + str(self.num_classes))
        logging.info("epoch_steps:%s" % self.epoch_steps)
        return train_loader, validate_loader, test_loader

    def create_backbone(self):
        ''' create backbone model

        :return: a backbone model
        '''    
        backbone = createBackbone(self.cfg, self.cfg.model_type, num_classes = self.num_classes, \
                    initial_pretrain=self.cfg.initial_pretrain, pretrain=self.cfg.pretrain)
        backbone.loss = MethodType(lambda obj,*args: self.loss(*args) ,backbone) # obj stands for backbone
        num_params = sum(param.numel() for param in backbone.parameters())
        self.backbone = backbone # may be use by other component
        logging.info('backbone:%s' % backbone)
        logging.info("Model params: %s" % num_params)
        print("Model params: ", num_params)
        return backbone

    def create_finetuner(self):
        ''' create finetuner

        :return a finetuner
        '''
        if self.cfg.finetuner.type == "":
            finetuner = None
        elif self.cfg.finetuner.type == "Basic":
            pretrained_model = createBackbone(self.cfg, self.cfg.model_type, num_classes=self.cfg.finetuner.pretrained_num_classes, \
                                initial_pretrain=self.cfg.finetuner.initial_pretrain, pretrain=self.cfg.finetuner.pretrain)
            finetuner = BasicFinetunner(pretrained_model, is_frozen=self.cfg.finetuner.frozen)
        else:
            logging.error("[%s] is not supported"%self.cfg.finetuner.type)
            raise NotImplementedError("[%s] is not supported"%self.cfg.finetuner.type)    
        self.finetuner = finetuner
        logging.info('finetuner:%s' % finetuner)
        return finetuner

    def create_distiller(self):
        ''' create distiller
        
        :return a distiller
        '''
        if self.cfg.distiller.type == "":
            distiller = None
        else:
            teacher_model = createBackbone(self.cfg, self.cfg.distiller.teacher.type, num_classes = self.num_classes, \
                                initial_pretrain=self.cfg.distiller.teacher.initial_pretrain, pretrain = self.cfg.distiller.teacher.pretrain)
            if self.cfg.distiller.type == "kd":
                distiller = KD(pretrained_model = teacher_model, 
                                temperature = self.cfg.kd.temperature,
                                is_frozen = self.cfg.distiller.teacher.frozen, 
                                use_saved_logits = self.cfg.distiller.use_saved_logits,
                                topk=self.cfg.distiller.logits_topk, 
                                num_classes=self.num_classes, 
                                teacher_type = self.cfg.distiller.teacher.type)
            elif self.cfg.distiller.type == "DKD":
                distiller = DKD(pretrained_model = teacher_model, 
                                alpha = self.cfg.dkd.alpha, beta = self.cfg.dkd.beta,
                                temperature = self.cfg.kd.temperature, warmup = self.cfg.dkd.warmup,
                                is_frozen = self.cfg.distiller.teacher.frozen, 
                                use_saved_logits = self.cfg.distiller.use_saved_logits,
                                topk=self.cfg.distiller.logits_topk, 
                                num_classes=self.num_classes, 
                                teacher_type = self.cfg.distiller.teacher.type)
            else:
                logging.error("[%s] is not supported"%self.cfg.distiller.type)
                raise NotImplementedError("[%s] is not supported"%self.cfg.distiller.type)
        self.distiller = distiller
        logging.info('distiller:%s' % distiller)
        return distiller

    def create_adapter(self):
        ''' create adapter

        :return an adapter
        '''
        if self.cfg.adapter.type == "":
            self.adapter = None
            return None
        
        if self.cfg.adapter.type == "CDAN":
            args_dict = {
                'in_feature': self.cfg.adapter.feature_size * self.num_classes, 'hidden_size': self.cfg.adapter.feature_size,
                'dropout_rate': 0.0, 'grl_coeff_alpha': 5.0, 'grl_coeff_high': 1.0, 'max_iter': self.epoch_steps, 'backbone_output_size': self.num_classes, 
                'enable_random_layer': False, 'enable_entropy_weight': False
            }
        elif self.cfg.adapter.type == "CAC_UNet":
            args_dict = {
                'input_channels': self.cfg.adapter.encoder_channels,
                'threeD': True,
                'pool_op_kernel_sizes': self.cfg.adapter.pool_op_kernel_sizes,
                'loss_weights': [self.cfg.adapter.encoder_weight, self.cfg.adapter.decoder_weight, 0]
            }
        else:
            logging.error("[%s] is not supported"%self.cfg.adapter.type)
            raise NotImplementedError("[%s] is not supported"%self.cfg.adapter.type)
        
        self.adapter = createAdapter(self.cfg.adapter.type, **args_dict)
        logging.info('adapter:%s' % self.adapter)
        return self.adapter

    def create_transfer_model(self, backbone, finetuner, distiller, adapter):
        ''' create transferrable model

        :return: a transferrable model
        '''
        if self.cfg.experiment.strategy == "":
            strategy = None
            model = backbone
        elif self.cfg.experiment.strategy == "OnlyFinetuneStrategy":
            strategy = TransferStrategy.OnlyFinetuneStrategy
            model = make_transferrable_with_finetune(backbone, self.loss, finetunner=finetuner)
        elif self.cfg.experiment.strategy == "OnlyDistillationStrategy":
            strategy = TransferStrategy.OnlyDistillationStrategy
            model = make_transferrable_with_knowledge_distillation(backbone,self.loss,distiller,
                                                   True,self.cfg.loss_weight.backbone,self.cfg.loss_weight.distiller)
        elif self.cfg.experiment.strategy == "OnlyDomainAdaptionStrategy":
            strategy = TransferStrategy.OnlyDomainAdaptionStrategy
            model = make_transferrable_with_domain_adaption(
                backbone, self.loss, adapter,
                False,
                self.cfg.loss_weight.backbone,
                self.cfg.loss_weight.adapter
            )
        elif self.cfg.experiment.strategy == "FinetuneAndDomainAdaptionStrategy":
            strategy = TransferStrategy.FinetuneAndDomainAdaptionStrategy
        elif self.cfg.experiment.strategy == "DistillationAndDomainAdaptionStrategy":
            strategy = TransferStrategy.DistillationAndDomainAdaptionStrategy
        else:
            raise NotImplementedError("[%s] is not supported"%self.cfg.experiment.strategy)
        
        self.model = model # may be use by other component
        return model, strategy

    def create_optimizer(self):
        ''' get optimizer

        :return: an optimizer
        '''
        if self.cfg.experiment.strategy == "OnlyFinetuneStrategy":
            finetuned_state_keys = ["backbone.%s"%name for name in self.finetuner.finetuned_state_keys] # add component prefix
            finetuner_params = {'params':[p for (name, p) in self.model.named_parameters() if p.requires_grad and name in finetuned_state_keys],
                                'lr': self.cfg.finetuner.finetuned_lr}
            remain_params = {'params':[p for (name, p) in self.model.named_parameters() if p.requires_grad and name not in finetuned_state_keys],
                                'lr': self.cfg.learning_rate}
            logging.info("[%s] params set finetuner finetuned learning rate[%s]" % (len(finetuner_params['params']), self.cfg.finetuner.finetuned_lr))
            logging.info("[%s] params set common learning rate [%s]" % (len(remain_params['params']), self.cfg.learning_rate))
            assert len(finetuner_params) > 0,"Empty finetuner_params"
            parameters = [finetuner_params,remain_params]
        else:
            remain_params = {'params': [p for p in self.model.parameters() if p.requires_grad],
                             'lr': self.cfg.learning_rate}
            logging.info("[%s] params set common learning rate [%s]" % (len(remain_params), self.cfg.learning_rate))
            parameters = [remain_params]
     
        optimizer = utils.create_optimizer(None, self.cfg, parameters)
        logging.info("optimizer:%s"%optimizer)
        return optimizer
        
    def run(self, eval=False, resume=False):
        ''' Run task
        :param eval: whether only for evaluate
        :param resume: whether resume for train
        Returns: validation metric.
        '''
        ######################### create components #########################
        ### dataloader
        train_loader, validate_loader, test_loader = self.create_dataloader()
        ### create distiller, check whether need to prepare logits
        distiller = self.create_distiller()
        if distiller is not None and (self.cfg.distiller.save_logits or self.cfg.distiller.check_logits):
            distiller.prepare_logits(train_loader, self.cfg.train_epochs,
                                        start_epoch=self.cfg.distiller.save_logits_start_epoch,
                                        device = self.device,
                                        save_flag=self.cfg.distiller.save_logits,
                                        check_flag=self.cfg.distiller.check_logits)
            return
        ### create Model Adapter components
        backbone = self.create_backbone()
        finetuner = self.create_finetuner()
        adapter = self.create_adapter()
        model,strategy = self.create_transfer_model(backbone, finetuner, distiller, adapter)
        is_transferrable = strategy != None 
        optimizer = self.create_optimizer()
        ### ipex wrapper 
        if "enable_ipex" in self.cfg and self.cfg.enable_ipex:
            import intel_extension_for_pytorch as ipex
            model = model.to(memory_format = torch.channels_last)
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            if is_transferrable: # ipex bug: some attribute is lost 
                setattr(model.backbone, "loss", self.loss)
        logging.info("model:%s"%model)
        ### create other training components
        metric = self.create_metric()
        lr_scheduler = utils.create_scheduler(optimizer, self.cfg)
        warmup_scheduler = utils.create_warmup_scheduler(optimizer,self.epoch_steps, self.cfg)
        tensorboard_writer = utils.create_tensorboard_writer(self.cfg)
        early_stopping = utils.create_early_stopping(self.cfg)
        profiler = utils.create_profiler(self.cfg)
        
        #################################### train and evaluate ###################
        ### define trainer
        trainer = TorchTrainerMA(cfg=self.cfg, model=model, metric=metric, best_metric_name=self.cfg.eval_metric, \
                        tensorboard_writer=tensorboard_writer, train_dataloader=train_loader, eval_dataloader=validate_loader, \
                        is_transferrable=is_transferrable, optimizer=optimizer, scheduler=lr_scheduler, warmup_scheduler=warmup_scheduler, \
                        criterion=None, early_stopping=early_stopping, profiler=profiler, device=self.device)
        logging.info("trainer:%s" % trainer)
        ### only evaluate
        if ext_dist.my_rank <= 0 and eval: # only non-distributed training, or rank 0 in distributed training
            with utils.Timer():
                start = time.time()
                metric_values = trainer.evaluate(epoch_steps=len(test_loader))
                total_seconds = time.time() - start
                test_num = len(test_loader.dataset)
                print(f"test data: {test_num}, throughput: {test_num/total_seconds} samples/second")
            return metric_values
        ### fit
        with utils.Timer():
            metric_values = trainer.fit(self.epoch_steps, self.model_dir, resume)
        return metric_values
