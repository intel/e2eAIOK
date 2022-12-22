import os
import sys
import time
import torch
import random
import logging
from tqdm import tqdm
from thop import profile

from e2eAIOK.DeNas.nlp.utils import customer_ops_map_thop
import e2eAIOK.common.trainer.utils.utils as utils
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer

class BERTTrainer(TorchTrainer):
    def __init__(self, cfg, model, train_dataloader, eval_dataloader, other_data, optimizer, criterion, scheduler, metric):
        super(BERTTrainer, self).__init__(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        self.other_data = other_data
        self.best_acc = -1
        self.is_stop = False

    def _pre_process(self):
        """
            trainer pre process to prepare trainer environment
        """
        utils.init_log()
        self.logger = logging.getLogger('Trainer')
        self.logger.info(f"Trainer config: {self.cfg}")
        self._dist_wrapper()
        if 'profile_flops' in self.cfg and self.cfg.profile_flops:
            inputs_id = torch.randint(1, 100, (1, self.cfg.max_seq_length, 1), dtype=torch.long)
            inputs_mask = torch.ones((1, self.cfg.max_seq_length, 1), dtype=torch.long)
            inputs_segment = torch.zeros((1, self.cfg.max_seq_length, 1), dtype=torch.long)
            inputs = torch.concat((inputs_id, inputs_mask, inputs_segment), -1)
            custom_ops_thop = customer_ops_map_thop()
            macs_thop, _ = profile(self.model, inputs=(inputs,), custom_ops=custom_ops_thop)
            logging.info("(THOP) MACs: %.2f" % (macs_thop/(1000**3)))

    def _is_early_stop(self, metric):
        return super()._is_early_stop(metric)

    def _dist_wrapper(self):
        """
            wrapper model for distributed training
        """
        if ext_dist.my_size > 1:
            self.model = ext_dist.DDP(self.model, find_unused_parameters=True)

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
        # random.seed(epoch)
        
        self.model.train()

        for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()       
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            if ('eval_step' in self.cfg and self.global_step % self.cfg.eval_step == 0) or self.global_step >= (self.cfg.num_train_steps * self.cfg.train_epochs):
                self.model.eval()
                result = self.evaluate(epoch)
                if result[self.cfg.eval_metric] > self.best_acc:
                    self.best_acc = result[self.cfg.eval_metric]
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    output_model_file = os.path.join(self.cfg.output_dir, self.cfg.task_name)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    self.other_data[-1].save_vocabulary(self.cfg.output_dir)
                if self.cfg.metric_threshold != 'None':
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