# coding=utf-8
# Copyright (c) 2022, Intel. and its affiliates.
# Copyright (c) 2021, Microsoft CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import ast
import time
import datetime
import yaml
import torch
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from timm.utils import accuracy
from collections import defaultdict, deque
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.optim.lr_scheduler import _LRScheduler

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64)
        if ext_dist.my_size > 1:
            ext_dist.dist.barrier()
            ext_dist.dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class EarlyStopping():
    ''' Early Stopping

    '''
    def __init__(self, tolerance_epoch = 5, delta=0, is_max = False, metric_threshold = None):
        ''' Init method

        :param tolerance_epoch: tolarance epoch
        :param delta: delta for difference
        :param is_max: max or min
        :param metric_threshold: when metric up/down limiation then stop training. None means ignore.
        '''

        self._tolerance_epoch = tolerance_epoch
        self._delta = delta
        self._is_max = is_max
        self._metric_threshold = metric_threshold

        self._counter = 0
        self.early_stop = False

    def __call__(self, validation_metric, optimal_metric):
        ############## absolute level #################
        if self._metric_threshold is not None:
            if (self._is_max and validation_metric >= self._metric_threshold) or\
            ((not self._is_max) and validation_metric <= self._metric_threshold):
                self.early_stop = True
                print("Earlystop when meet metric_threshold [%s]"%self._metric_threshold)
                logging.info("Earlystop when meet metric_threshold [%s]"%self._metric_threshold)
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
        _str += '\tmetric_threshold:%s\n' % self._metric_threshold
        return _str

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

def get_device(cfg):
    if 'device' in cfg and cfg.device in ['gpu', 'cuda'] and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def is_main_process():
    if ext_dist.my_size > 1:
        return ext_dist.dist.get_rank() == 0
    return 0

def save_model(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_log():
    if ext_dist.my_size > 1:
        if ext_dist.my_rank == 0:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)


def acc(output,label):
    pred = output.data.cpu().max(1)[1]
    label = label.data.cpu()
    if label.shape == output.shape:
        label = label.max(1)[1]

    if pred.shape != label.shape:
        logging.error('pred shape[%s] and label shape[%s] not match' % (pred.shape, label.shape))
        raise RuntimeError('pred shape[%s] and label shape[%s] not match' % (pred.shape, label.shape))
    return torch.mean((pred == label).float())

def create_metric(cfg):
    if cfg.eval_metric == "accuracy":
        metric = accuracy
    return metric

def create_criterion(cfg):
    if cfg.criterion == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        logging.error("[%s] is not supported"%cfg.criterion)
        raise NotImplementedError("[%s] is not supported"%cfg.criterion)
    return criterion

def create_optimizer(model=None, cfg=None, parameters=None):
    ''' create optimizer

    :param model: model, will be disabled if 'parameters' is not None
    :param cfg: configurations
    :param parameters: parameters for optimizer

    :return: a optimizer
    '''
    logging.info(F"model:{model}")
    logging.info(F"parameters:{parameters}")
    parameters = parameters if parameters is not None else model.parameters()
    if cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=cfg.learning_rate,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        logging.error("[%s] is not supported"%cfg.optimizer)
        raise NotImplementedError("[%s] is not supported"%cfg.optimizer)
    return optimizer

def create_scheduler(optimizer, cfg):
    ''' create scheduler

    :return: a scheduler
    '''
    if "lr_scheduler" not in cfg or cfg.lr_scheduler == "":
        scheduler = None
    elif cfg.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_scheduler_config.decay_rate)
    elif cfg.lr_scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_scheduler_config.decay_stages, gamma=cfg.lr_scheduler_config.decay_rate)
    elif cfg.lr_scheduler == "CosineAnnealingLR":
        t_max = cfg.lr_scheduler.T_max if "T_max" in cfg.lr_scheduler and cfg.lr_scheduler.T_max > 0 else cfg.train_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif cfg.lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",factor=cfg.lr_scheduler_config.decay_rate, patience=cfg.lr_scheduler_config.decay_patience, \
                                                        verbose=True, threshold_mode='abs')
    else:
        logging.error("[%s] is not supported"%cfg.lr_scheduler)
        raise NotImplementedError("[%s] is not supported"%cfg.lr_scheduler)
    return scheduler

def create_warmup_scheduler(optimizer, total_iters, cfg):
    ''' create warm up scheduler

    :param optimizer: optimizer
    :param total_iters: iterations in one epoch
    :return: a warm up scheduler
    '''
    if "warmup_scheduler" not in cfg or cfg.warmup_scheduler == "" or \
        "warmup_scheduler_epoch" not in cfg or cfg.warmup_scheduler_epoch <= 0:
        warmup_scheduler = None
    elif cfg.warmup_scheduler == "WarmUpLR":
        warmup_scheduler = WarmUpLR(optimizer, total_iters * cfg.warmup_scheduler_epoch)
    else:
        logging.error("[%s] is not supported"%cfg.warmup_scheduler)
        raise NotImplementedError("[%s] is not supported"%cfg.warmup_scheduler)
    
    return warmup_scheduler

def create_early_stopping(cfg):
    ''' create early_stopping

    :return: an early_stopping
    '''
    if "early_stop" not in cfg or cfg.early_stop == "":
        early_stopping = None
    elif cfg.early_stop == "EarlyStopping":
        early_stopping = EarlyStopping(tolerance_epoch=cfg.early_stop_config.tolerance_epoch, delta=cfg.early_stop_config.delta, 
                                    is_max=cfg.early_stop_config.is_max, metric_threshold=cfg.metric_threshold)
        logging.info('early_stopping :%s' % early_stopping)
    else:
        logging.error("[%s] is not supported"%cfg.early_stop)
        raise NotImplementedError("[%s] is not supported"%cfg.early_stop)
    return early_stopping

def create_tensorboard_writer(cfg):
    ''' create tensorboard_writer

    :return: a tensorboard_writer
    '''
    if "tensorboard_dir" not in cfg or cfg.tensorboard_dir == "":
        tensorboard_writer  = None 
    else:
        tensorboard_writer = SummaryWriter(cfg.tensorboard_dir)
        logging.info('tensorboard_writer :%s' % tensorboard_writer)
    return tensorboard_writer

def trace_handler(trace_file,p):
    ''' profile trace handler

    :param trace_file: output trace file
    :param p: profiler
    :return:
    '''
    logging.info("trace_output:%s" % p.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    p.export_chrome_trace(trace_file + str(p.step_num) + ".json")

def create_profiler(cfg):
    ''' create a profiler

    :return: profiler
    '''
    if "profiler" not in cfg or not cfg.profiler:
        return None
    def parse_activities(activity_str):
        result = set()
        for item in activity_str.lower().split(","):
            if item == 'cpu':
                result.add(ProfilerActivity.CPU)
            elif item == 'gpu' or item == 'cuda':
                result.add(ProfilerActivity.CUDA)
        return result

    activities = parse_activities(get_device(cfg))
    schedule = torch.profiler.schedule(skip_first=cfg.profiler_config.skip_first,wait=cfg.profiler_config.wait,
        warmup=cfg.profiler_config.warmup,active=cfg.profiler_config.active,repeat=cfg.profiler_config.repeat)
    profiler = profile(activities=activities,schedule=schedule,
                        on_trace_ready=partial(trace_handler,cfg.profiler_config.trace_file))
    logging.info("profiler:%s"%profiler)
    return profiler