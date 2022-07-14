#!/usr/bin/python
# -*- coding: utf-8 -*-
import os.path

import torch
from .backbone.lenet import LeNet
import logging

class ModelManager:
    ''' Manage target models

    '''
    def __init__(self,path):
        ''' Init method

        :param path: path to store target models
        '''
        self._path = path.strip().rstrip("/")
        self.model_names = {}
        self._model_names_filepath = "%s/model_names.txt" % self._path
        self._loadModelNames()

    def _loadModelNames(self):
        ''' load the model_name and class_name of stored target models

        :return:
        '''
        if os.path.exists(self._model_names_filepath):
            with open(self._model_names_filepath) as f:
                for line in f:
                    line = line.rstrip("\n")
                    parts = line.split("\t")
                    model_name = parts[0]
                    model_class = parts[1]
                    self.model_names[model_name] = model_class

    def _saveModelNames(self):
        ''' save the model_name and class_name of stored target models

        :return:
        '''
        with open(self._model_names_filepath,'w') as f:
            for (model_name,model_class) in sorted(self.model_names.items(),key=lambda x:x[0]):
                f.write("%s\t%s\n"%(model_name,model_class))

    def getModelPath(self,model_name):
        ''' Given the model name, return the model path

        :param model_name: model name
        :return: model path
        '''
        return "%s/model_%s"%(self._path,model_name)

    def register(self,model_name,model):
        ''' resiter the given model with model name, then store it

        :param model_name: model name
        :param model: trained model object
        :return:
        '''
        if model_name in self.model_names:
            logging.error("model name [%s] has been registered!"%model_name)
            raise RuntimeError("model name [%s] has been registered!"%model_name)
        else:
            torch.save(model,self.getModelPath(model_name))
            self.model_names[model_name] = type(model).__name__
            self._saveModelNames()
            logging.info("register model [%s]"%model_name)

    def load(self,model_name):
        ''' Given the model name, return the model object

        :param model_name:
        :return: model object
        '''
        model_path = self.getModelPath(model_name)
        if os.path.exists(model_path):
            return torch.load(model_path)
        else:
            logging.error("path [%s] of model name [%s] not exist!"%(model_path,model_name))
            raise RuntimeError("path [%s] of model name [%s] not exist!"%(model_path,model_name))

