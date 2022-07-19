#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.optim as optim
import torch
import torch.nn as nn
import logging
from .utils import init_weights,EarlyStopping
from .backbone.utils import copyParameterFromPretrained
import time
import datetime

class TLEngine:
    ''' Transfer Learning Engine

    '''
    def __init__(self, task_manager,model_manager, backbone,discriminator,tensorboard_writer):
        ''' Init method

        :param task_manager: task manager to manage transfer learning task
        :param model_manager: model manager to  manage learned model
        :param backbone: backbone model
        :param discriminator: discriminator model
        :param tensorboard_writer: tensorboard writer
        '''
        self._task_manager = task_manager
        self._model_manager = model_manager
        self._backbone = backbone
        self._discriminator = discriminator
        self._tensorboard_writer = tensorboard_writer
        self._early_stopping = EarlyStopping(tolerance_epoch=task_manager.earlystopping_tolerance,
                                             delta=task_manager.earlystopping_delta,is_max=True)

        self._initializeModel(task_manager.backbone_pretrained,
                              task_manager.pretrained_path,
                              task_manager.pretrained_layer_pattern)
        self._createOptimizer()
    def _initializeModel(self,backbone_pretrained = False,pretrained_path=None,pretrained_layer_pattern=None):
        ''' initialize model

        :param backbone_pretrained: whether use pretraining
        :param pretrained_path: where to load pretrained model
        :param pretrained_layer_pattern: which layer of backbone should be pretrained
        :return:
        '''
        self._backbone.apply(init_weights)
        self._discriminator.apply(init_weights)
        if backbone_pretrained and pretrained_path and pretrained_layer_pattern:
            copyParameterFromPretrained(self._backbone, torch.load(pretrained_path), pretrained_layer_pattern)
            logging.info("Backbone pretraining")

    def _createOptimizer(self):
        ''' create optimizer for backbone and discriminator

        :return:
        '''
        self._backbone_optimizer = optim.SGD(self._backbone.parameters(),
                                    lr=self._task_manager.traing_backbone_lr,
                                    weight_decay=self._task_manager.traing_backbone_weight_decay,
                                    momentum=self._task_manager.traing_backbone_momentum)
        self._discriminator_optimizer = optim.SGD(self._discriminator.parameters(),
                                       lr=self._task_manager.traing_discriminator_lr,
                                       weight_decay=self._task_manager.traing_discriminator_weight_decay,
                                       momentum=self._task_manager.traing_discriminator_momentum)
    def _initializeOptimizer(self):
        ''' initialize optimizer

        :return:
        '''
        self._backbone_optimizer.zero_grad()
        self._discriminator_optimizer.zero_grad()
    def _stepOptimizer(self,current_epoch):
        ''' step optimizer

        :param current_epoch:
        :return:
        '''
        self._backbone_optimizer.step()
        if current_epoch > 1:
            self._discriminator_optimizer.step()

    def _train_epoch(self,source_train_dataloader,target_train_dataloader,current_epoch,train_source_iter_num
                     ,train_target_iter_num,total_epoch):
        ''' train epoch

        :param source_train_dataloader: source train dataloader
        :param target_train_dataloader: target train dataloader
        :param current_epoch: current epoch
        :param train_source_iter_num: training source iter num per epoch
        :param train_target_iter_num: training target iter num per epoch
        :param total_epoch: total epoch
        :return:
        '''
        self._backbone.train()      # set training flag
        self._discriminator.train() # set training flag

        num_iter_per_epoch = max(train_source_iter_num ,train_target_iter_num) # num iter per epoch is the max one

        for batch_idx in range(num_iter_per_epoch):
            global_iter_idx = batch_idx + current_epoch * num_iter_per_epoch # global iter index

            if batch_idx % train_source_iter_num == 0: # end of train source
                logging.info("reset source_train_dataloader")
                iter_source = iter(source_train_dataloader)
            if batch_idx % train_target_iter_num == 0: # end of train target
                logging.info("reset target_train_dataloader")
                iter_target = iter(target_train_dataloader)
            data_source, label_source = iter_source.next()
            # data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target, label_target = iter_target.next()
            #data_target = data_target.cuda()
            if data_source.numel() ==0 or label_source.numel() ==0 or \
                    data_target.numel() ==0 or label_target.numel() ==0:
                logging.warning('empty batch at epoch[%s] index[%s]: data_source batch [%s],label_source batch [%s],'
                                'data_target batch [%s], label_target batch [%s]'%(
                    current_epoch,batch_idx,data_source.numel(),label_source.numel(),
                    data_target.numel(),label_target.numel()))
                continue
            self._initializeOptimizer()

            output_logit_source, discriminator_feature_source = self._backbone(data_source)
            output_logit_target, discriminator_feature_target = self._backbone(data_target)

            backbone_loss = self._backbone.loss(output_logit_source,label_source,self._tensorboard_writer)
            if self._task_manager.traning_enable_target_label:
                logging.info("enable training with target label")
                backbone_loss += self._backbone.loss(output_logit_target,label_target,self._tensorboard_writer)

            discriminator_loss =  self._discriminator.loss(discriminator_feature_source,output_logit_source,label_source,
                                                         discriminator_feature_target,output_logit_target,label_target,
                                                         self._tensorboard_writer)
            loss = backbone_loss + discriminator_loss
            loss.backward()
            self._stepOptimizer(current_epoch)

            self._tensorboard_writer.add_scalar('Loss/train', loss.item(), global_iter_idx)
            self._tensorboard_writer.add_scalar('Loss/train_backbone', backbone_loss.item(),global_iter_idx)
            self._tensorboard_writer.add_scalar('Loss/train_discriminator', discriminator_loss.item(),global_iter_idx)

            self._tensorboard_writer.add_histogram('backbone_logit_out/source', output_logit_source,global_iter_idx)
            self._tensorboard_writer.add_histogram('backbone_logit_out/target', output_logit_target,global_iter_idx)
            self._tensorboard_writer.add_histogram('discriminator_feature/source', discriminator_feature_source,global_iter_idx)
            self._tensorboard_writer.add_histogram('discriminator_feature/target', discriminator_feature_target,global_iter_idx)

            if (global_iter_idx) % self._task_manager.training_log_interval == 0:
                out_str = '%s Train Epoch: %s/%s [%s/%s (%0.1f)]\tLoss: %0.3f\tBackbone Loss:%0.3f\tDiscriminator Loss:%0.3f'%(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),current_epoch,total_epoch,
                    batch_idx * self._task_manager.batch_size,
                    num_iter_per_epoch * self._task_manager.batch_size,
                    100. * batch_idx / num_iter_per_epoch,
                    loss.item(),backbone_loss.item(),discriminator_loss.item())
                print(out_str)
                logging.info(out_str)

    def _evaluate_epoch(self, model,target_dataloader,current_epoch,is_test):
        ''' evaluate epoch

        :param model: model
        :param target_dataloader: target dataloader
        :param current_epoch: current epoch
        :param is_test: is test
        :return:
        '''
        with torch.no_grad():
            model.eval()       # set evaluating flag

            test_loss = 0
            correct = 0
            sample_num = 0
            for data, label in target_dataloader:
                # data, target = data.cuda(), target.cuda()
                output,_ = model(data)
                sample_num += data.size(0)
                test_loss += nn.CrossEntropyLoss()(output, label).item()
                pred = output.data.cpu().max(1)[1]
                label = label.data.cpu()
                if pred.shape != label.shape:
                    logging.error('pred shape[%s] and label shape[%s] not match'%(pred.shape,label.shape))
                    raise RuntimeError('pred shape[%s] and label shape[%s] not match'%(pred.shape,label.shape))
                correct += pred.eq(label).sum().item()

            test_loss /= sample_num
            test_acc = 100.*correct/sample_num
            prefix = 'test' if is_test else 'evaluate'
            self._tensorboard_writer.add_scalar('Loss/%s_target_loss'%prefix, test_loss, current_epoch)
            self._tensorboard_writer.add_scalar('Loss/%s_target_acc'%prefix, test_acc, current_epoch)

            out_str = '\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                prefix,test_loss, correct, sample_num, test_acc)
            print(out_str)
            logging.info(out_str)
            return test_loss,test_acc

    def train(self, source_train_dataloader, target_train_dataloader, target_valid_dataloader,
              train_source_iter_num ,train_target_iter_num):
        ''' transfer learning train

        :param source_train_dataloader: source train dataloader
        :param target_train_dataloader: target train dataloader
        :param target_valid_dataloader: target validate dataloader
        :param train_source_iter_num: train source iter num per epoch
        :param train_target_iter_num: train target iter num per epoch
        :return: name of the trained model
        '''
        for epoch in range(1, self._task_manager.traing_epochs + 1):
            self._train_epoch(source_train_dataloader, target_train_dataloader, epoch,
                              train_source_iter_num ,train_target_iter_num,
                              self._task_manager.traing_epochs)
            test_loss,test_acc = self._evaluate_epoch(self._backbone,target_valid_dataloader,epoch,False)
            self._early_stopping(test_acc,self._backbone.state_dict())
            self._tensorboard_writer.flush()
            if self._early_stopping.early_stop:
                logging.warning("Early stop after epoch:%s, the best acc is %s"%(epoch,
                                                                        self._early_stopping.optimal_metric))
                break
        if self._early_stopping.optimal_model is not None:
            self._backbone.load_state_dict(self._early_stopping.optimal_model)

        model_name = "%s_%s_%s"%(self._backbone.__class__.__name__,self._discriminator.__class__.__name__,int(time.time()))

        self._model_manager.register(model_name,self._backbone)
        return model_name

    def evaluate(self,model_name,target_test_loader):
        ''' transfer learning evaluate

        :param model_name: the model name
        :param target_test_loader: target test loader
        :return:
        '''
        model = self._model_manager.load(model_name)
        self._evaluate_epoch(model,target_test_loader,1,True)