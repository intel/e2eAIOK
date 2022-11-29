import sys
import time
import e2eAIOK.common.trainer.utils.utils as utils
import random
import torch
import logging
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from abc import ABC, abstractmethod
from e2eAIOK.common.trainer.data_builder import DataBuilder
from e2eAIOK.common.trainer.model_builder import ModelBuilder
 
class TorchTrainer():
    """
    The basic trainer class for all models

    Note:
        You should implement specfic model trainer under model folder like vit_trainer
    """
    def __init__(self, cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.metric = metric
        
    def _pre_process(self):
        """
            trainer pre process to prepare trainer environment
        """
        utils.init_log()
        self.logger = logging.getLogger('Trainer')
        self.logger.info(f"Trainer config: {self.cfg}")
        self._dist_wrapper()

    def _post_process(self):
        """
            trainer post process
        """
        self.logger.info("Trainer complete")
        
    def _dist_wrapper(self):
        """
            wrapper model for distributed training
        """
        if ext_dist.my_size > 1:
            self.model = ext_dist.DDP(self.model)
    
    def _is_early_stop(self, metric):
        """
            check whether training achieved pre-defined metric threshold
        """
        return metric >= self.cfg["metric_threshold"]

    def train_one_epoch(self, epoch):
        """
            train one epoch
        """
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        self.model.train()

        for inputs, targets in self.train_dataloader:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss_value = loss.item()
            self.optimizer.zero_grad()       
            loss.backward()
            self.optimizer.step()
        
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            self.global_step += 1

            if 'eval_step' in self.cfg and self.global_step % self.cfg.eval_step == 0:
                self.model.eval()
                metric = utils.create_metric(self.cfg ,self.model, self.data_loader)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info("Averaged stats:", metric_logger)

    def evaluate(self, epoch):
        """
            evaluatiuon
        """
        metric_logger = utils.MetricLogger(delimiter="")

        self.model.eval()
        
        for inputs, target in self.eval_dataloader:
            output = self.model(inputs)
            loss = self.criterion(output, target)
            metric = self.metric(output, target)

            metric_logger.update(loss=loss.item())
            for k in metric.keys():
                metric_logger.meters[k].update(metric[k], n=self.cfg.eval_batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        output_str = []
        for name, meter in metric_logger.meters.items():
            output_str.append(
                "{}: {}".format(name, str(meter))
            )
        self.logger.info(output_str)

        if self.cfg.output_dir:
            checkpoint_paths = [self.cfg.output_dir + '/checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_model({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

    def fit(self):
        """
            trainint and evaluation
        """
        self._pre_process()
        start_time = time.time()
        for i in range(1, self.cfg.train_epochs):
            train_start = time.time()
            self.train_one_epoch(i)
            if i % self.cfg.eval_epochs == 0:
                eval_start = time.time()
                metric = self.evaluate(i)
                self.logger.info(F"Evaluate time:{time.time() - eval_start}")
                if self._is_early_stop(metric):
                    self.logger.info(f"Metric {metric} got threshold {self.cfg['metric_threshold']}, early stop")
                    break
            self.logger.info(F"Epoch {i} training time:{time.time() - train_start}")

        self.logger.info(F"Total time:{time.time() - start_time}")
        self._post_process()
