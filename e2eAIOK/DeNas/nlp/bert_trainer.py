import os
import sys
import time
from tqdm import tqdm
import torch
import random

from e2eAIOK.common.trainer.torch_trainer import TorchTrainer

class BERTTrainer(TorchTrainer):
    def __init__(self, cfg, model, train_dataloader, eval_dataloader, other_data, optimizer, criterion, scheduler, metric):
        super(BERTTrainer, self).__init__(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        self.other_data = other_data
        self.best_acc = -1
        self.is_stop = False

    def _is_early_stop(self, metric):
        return super()._is_early_stop(metric)

    def _post_process(self):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        parameter_size = model_to_save.calc_sampled_param_num()
        output_str = "**************S*************\n" + \
                             "task_name = {}\n".format(self.cfg.task_name) + \
                             "parameter size = {}\n".format(parameter_size) + \
                             "total training time = {}\n".format(str(time.time() - self.start_time)) + \
                             "best_acc = {}\n".format(self.best_acc) + \
                             "**************E*************\n"
        self.logger.info(output_str)

    def train_one_epoch(self, epoch):
        # set random seed
        random.seed(epoch)
        
        self.model.train()

        for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()       
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            if 'eval_step' in self.cfg and self.global_step % self.cfg.eval_step == 0:
                self.model.eval()
                result = self.evaluate(epoch)
                if result[self.cfg.eval_metric] > self.best_acc:
                    self.best_acc = result[self.cfg.eval_metric]
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    output_model_file = os.path.join(self.cfg.output_dir, self.cfg.task_name)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    self.other_data[-1].save_vocabulary(self.cfg.output_dir)
                self.is_stop = self._is_early_stop(result[self.cfg.eval_metric])
                if self.is_stop:
                    break

    def evaluate(self, epoch):
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Epoch = {} iter {} step".format(epoch, self.global_step))
        result = self.metric(self.cfg, self.model, self.eval_dataloader, self.other_data)
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("{} = {}".format(key, str(result[key])))
        return result         

    def fit(self):
        self._pre_process()
        self.start_time = time.time()
        self.global_step = 0
        for i in range(self.cfg.train_epochs):
            train_start = time.time()
            self.train_one_epoch(i)
            self.logger.info(F"Epoch {i} training time:{time.time() - train_start}")
            if self.is_stop:
                break
        self._post_process()