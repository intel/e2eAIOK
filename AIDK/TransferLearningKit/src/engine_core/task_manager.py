#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os.path
import time
from bs4 import BeautifulSoup
from .backbone.factory import createBackbone
from .discriminator.factory import createDiscriminator
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset.image_list import ImageList
from dataset.office31 import Office31
from torchvision import transforms

class TaskManager:
    ''' TaskManager, could be loaded from server progress(for example, multi dataloader)

    '''
    def __init__(self,task_config_file):
        ''' Init method

        :param task_config_file: task config file (xml format)
        '''
        self._str = "task manager constructed from [%s]:" % task_config_file # for __str__()

        self._filename = task_config_file
        with open(task_config_file) as f:
            text = "\n".join(f.readlines())
            self._bs = BeautifulSoup(text,features="xml")
        self._configLogging()
        self._parseGlobal(self._bs.find("global"))
        self._parseTask(self._bs.find("task"))
        self._parseTraining(self._bs.find("training"))


    def _parseTraining(self,bs):
        ''' parse training config

        :param bs: bs segment which containing training details
        :return:
        '''
        self.traing_backbone_lr = float(bs.find('backbone_lr').string)
        self.traing_backbone_weight_decay = float(bs.find('backbone_weight_decay').string)
        self.traing_backbone_momentum = float(bs.find('backbone_momentum').string)

        self.traing_discriminator_lr = float(bs.find('discriminator_lr').string)
        self.traing_discriminator_weight_decay = float(bs.find('discriminator_weight_decay').string)
        self.traing_discriminator_momentum = float(bs.find('discriminator_momentum').string)

        self.traing_epochs = int(bs.find('epochs').string)
        self.training_log_interval = int(bs.find('log_interval').string)
        self.traning_enable_target_label = int(bs.find('target_label_enable').string) > 0
        self.earlystopping_tolerance = int(bs.find('earlystopping_tolerance').string)
        self.earlystopping_delta = float(bs.find('earlystopping_delta').string)

        self._str += '\n\tTraining:traing_backbone_lr[%s],' \
                     'traing_backbone_weight_decay[%s],' \
                     'traing_backbone_momentum[%s],' \
                     'traing_discriminator_lr[%s],' \
                     'traing_discriminator_weight_decay[%s],' \
                     'traing_discriminator_momentum[%s],' \
                     'traing_epochs[%s],' \
                     'training_log_interval[%s],' \
                     'traning_enable_target_label[%s],' \
                     'earlystopping_tolerance[%s],' \
                     'earlystopping_delta[%s]'%(
            self.traing_backbone_lr,
            self.traing_backbone_weight_decay,
            self.traing_backbone_momentum,
            self.traing_discriminator_lr,
            self.traing_discriminator_weight_decay,
            self.traing_discriminator_momentum,
            self.traing_epochs,
            self.training_log_interval,
            self.traning_enable_target_label,
            self.earlystopping_tolerance,
            self.earlystopping_delta)

    def _parseGlobal(self,bs):
        ''' parse global config

        :param bs: bs segment which containing global setting
        :return:
        '''
        self.seed = int(bs.find('seed').string)
        self.model_dir = bs.find('model_dir').string
        if not os.path.exists(self.model_dir):
            logging.warning("%s not exists, and create"%self.model_dir)
            os.makedirs(self.model_dir)

        self._str += '\n\tGlobal:seed[%s],model_dir[%s]'%(self.seed,self.model_dir)

    def _parseTask(self,bs):
        ''' parse task config

        :param bs: bs segment which containing task-specific setting
        :return:
        '''
        self.task_classificaton_num_class = int(bs.find('classification').find('num_classes').string)
        self._str += '\n\tTask:task_classificaton_num_class[%s]' % self.task_classificaton_num_class

    def createDiscriminator(self,num_iter_per_epoch):
        ''' create discriminator

        :param num_iter_per_epoch: num iter per epoch
        :param tensorboard_writer: tensorboard writer
        :return:
        '''
        node = self._bs.find("discriminator").find("adversarial") # the first one
        kwargs = {'max_iter':num_iter_per_epoch}
        for child in node.children:
            if child.name:
                kwargs[child.name] = child.text

        discriminator = createDiscriminator(node.attrs['name'], **kwargs)
        self._str += '\n\tDiscriminator: %s' % discriminator
        return discriminator

    def createBackbone(self):
        ''' create backbone

        :return: a backbone model
        '''
        node = self._bs.find("backbone").find("predefined")
        pretrained = node.find('pretrained')
        kwargs = {'num_classes': self.task_classificaton_num_class}
        if pretrained:
            self.backbone_pretrained = True
            self.pretrained_path = pretrained.find('path').string
            self.pretrained_layer_pattern = [item for item in pretrained.find('layer_pattern').string.strip().split("\n") if item]
            self._str += '\n\t Backbone Pretrained:[%s], path [%s], layer pattern [%s]'%(self.backbone_pretrained,
                                                                                         self.pretrained_path,
                                                                                         self.pretrained_layer_pattern)
        else:
            self.backbone_pretrained = False
            self._str += '\n\t Backbone Pretrained:[%s]' % (self.backbone_pretrained)

        if node:
            backbone =  createBackbone(node.attrs['name'],**kwargs)
            self._str += '\n\tBackbone: %s' % (backbone)
            return backbone
        else:  # customized
            logging.error("Create backbone failed!")
            raise RuntimeError("Create backbone failed!")

    def _configLogging(self):
        ''' Config logging system

        :return:
        '''
        dir_path = self._bs.find("global").find('logging').find('dir').string
        if not os.path.exists(dir_path):
            print("%s not exists, and create"%dir_path)
            os.makedirs(dir_path)

        filename = "%s/%s.log"%(dir_path,int(time.time()))
        level = self._bs.find("global").find('logging').find('level').string.upper()
        logging.basicConfig(filename=filename, level=logging._nameToLevel[level],
                            format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filemode='w')
        self._str += '\n\tLogging: log name[%s], log level[%s]'%(filename,level)

    def createTensorboardWriter(self):
        ''' create tensorboard writer

        :return: a SummaryWriter
        '''
        path = self._bs.find("global").find('tensorboard').find('dir').string
        if not os.path.exists(path):
            logging.warning("%s not exists, and create"%path)
            os.makedirs(path)

        backbone_name = self._bs.find("backbone").find("predefined").attrs['name']
        discriminator_name = self._bs.find("discriminator").find("adversarial").attrs['name']
        filename_suffix = "_%s_%s"%(backbone_name,discriminator_name)
        self._str += '\n\tTensorboardWriter: path[%s], filename_suffix[%s]' %(path,filename_suffix)
        return SummaryWriter(path,filename_suffix=filename_suffix)

    def createDatasets(self):
        datasets = {}
        bs = self._bs.find('datasets')
        self.batch_size = int(bs.find("batch_size").string)
        self.num_worker = int(bs.find("num_worker").string)

        self._str += '\n\tDatasets: batch size[%s], num worker [%s]' % (self.batch_size,self.num_worker)

        for seg in bs.find_all('dataset'):
            formatter = seg.attrs['formatter']
            train_type = seg.attrs['type']
            data_path = seg.find('data').string
            kwargs = {"data_transform": eval(seg.find('transform').string.strip()),
                      "img_mode": seg.find('img_mode').string}

            if formatter == 'ImageList':
                label_path = seg.find('label').string
                kwargs['label_records'] = open(label_path).readlines()
                dataset = ImageList(data_path,**kwargs)
            elif formatter == 'Office31':
                label_path = seg.find('label').string
                kwargs['label_map'] = {item.split('\t')[0]:int(item.split('\t')[1])
                                       for item in open(label_path).readlines()}
                dataset = Office31(data_path,**kwargs)
            else:
                logging.error("unknown formatter [%s]"%formatter)
                raise RuntimeError("unknown formatter [%s]"%formatter)
            datasets[train_type] =  dataset
            self._str += '\n\t\tDataset [%s] : %s'%(train_type,dataset)
        return datasets

    def __str__(self):
        return self._str