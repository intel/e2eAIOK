#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/27/2022 3:13 PM

import torch.nn as nn
import torch
import logging
import datetime
import os
import numpy as np

class EarlyStopping():
    ''' Early Stopping

    '''
    def __init__(self, tolerance_epoch = 5, delta=0, is_max = False, limitation = None):
        ''' Init method

        :param tolerance_epoch: tolarance epoch
        :param delta: delta for difference
        :param is_max: max or min
        :param limitation: when metric up/down limiation then stop training. None means ignore.
        '''

        self._tolerance_epoch = tolerance_epoch
        self._delta = delta
        self._is_max = is_max
        self._limitation = limitation

        self._counter = 0
        self.early_stop = False

    def __call__(self, validation_metric, optimal_metric):
        ############## absolute level #################
        if self._limitation is not None:
            if (self._is_max and validation_metric >= self._limitation) or\
            ((not self._is_max) and validation_metric <= self._limitation):
                self.early_stop = True
                print("Earlystop when meet limitation [%s]"%self._limitation)
                logging.info("Earlystop when meet limitation [%s]"%self._limitation)
                return
        ############## relative level #################
        if (self._is_max and (validation_metric < optimal_metric - self._delta)) \
                or ((not self._is_max) and (validation_metric > optimal_metric + self._delta)): # less optimal
            self._counter += 1
        else: # more optimal
            logging.info("Reset earlystop counter")
            self._counter = 0
        if self._counter >= self._tolerance_epoch:
            self.early_stop = True

    def __str__(self):
        _str = 'EarlyStopping:%s\n'%self._tolerance_epoch
        _str += '\tdelta:%s\n'%self._delta
        _str += '\tis_max:%s\n' % self._is_max
        _str += '\tlimitation:%s\n' % self._limitation
        return _str

from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class Timer:
    ''' Timer to stat elapsed time

    '''
    def __enter__(self):
        self.start = datetime.datetime.now()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = datetime.datetime.now()
        total_seconds = (self.end - self.start).total_seconds()
        _str = "Total seconds:%s" % (total_seconds)
        print(_str)
        logging.info(_str)