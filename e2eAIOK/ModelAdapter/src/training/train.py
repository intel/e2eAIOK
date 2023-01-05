import torch
import datetime, time
import contextlib
import os
import numpy as np
from collections.abc import Iterable
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from types import MethodType

class TorchTrainerMA(TorchTrainer):
    ''' Trainer for Model Adapter

    '''
    def __init__(self, cfg, model, metric, best_metric_name, tensorboard_writer=None, 
                    train_dataloader=None, eval_dataloader=None, is_transferrable=False,
                    optimizer=None, scheduler=None, warmup_scheduler=None, 
                    criterion=None, early_stopping=None, 
                    profiler=None, device='cpu'):
        ''' Init method
        :param cfg: configurations
        :param model: the trained model
        :param metric: metric function map, which map metric name to metric function
        :param best_metric_name: metric name for update best model
        :param tensorboard_writer: tensorboard writer, can be None
        :param train_dataloader: train dataloader, can be None in evaluate mode
        :param eval_dataloader: validation dataloader
        :param is_transferrable: is model transferrable
        :param optimizer: optimizer
        :param scheduler: learning rate scheduler, can be None
        :param warmup_scheduler: the scheduler for warmup, can be None
        :param criterion: criterion for loss function, can be None
        :param early_stopping: for early stopping, can be None
        :param profiler : training profiler
        :param device: running on cpu or gpu
        '''
        super().__init__(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        self.best_metric_name = best_metric_name
        self.tensorboard_writer = tensorboard_writer
        self.warmup_scheduler = warmup_scheduler
        self.early_stopping = early_stopping
        self.is_transferrable = is_transferrable
        self.profiler = profiler
        self.device = device
        self.best_metric_value = 0.0
        self.backbone = self.model.get_backbone() if self.is_transferrable else self.model
        if best_metric_name not in metric:
            raise RuntimeError("early stop metric [%s] not in metric keys [%s]"%(
                best_metric_name,",".join(metric.keys())
            ))

    def __str__(self):
        _str = "Trainer: model:%s\n"%self.model
        _str += "\tconfigurations:%s\n" % self.cfg
        _str += "\ttrain_dataloader:%s\n"%self.train_dataloader
        _str += "\teval_dataloader:%s\n"%self.eval_dataloader
        _str += "\toptimizer:%s\n"%self.optimizer
        _str += "\tscheduler:%s\n" % self.scheduler
        _str += "\tmetric:%s\n" % self.metric
        _str += "\tcriterion:%s\n" % self.criterion
        _str += "\t_best_metric_name:%s\n" % self.best_metric_name
        _str += "\ttensorboard_writer:%s\n" % self.tensorboard_writer
        _str += "\twarmup_scheduler:%s\n" % self.warmup_scheduler
        _str += "\tearly_stopping:%s\n" % self.early_stopping
        _str += "\tis_transferrable:%s\n" % self.is_transferrable
        _str += "\tprofiler:%s\n" % self.profiler
        _str += "\tdevice:%s\n" % self.device
        _str += "\trank:%s\n" % ext_dist.my_rank
        _str += "\tbest_metric_value:%s\n" % self.best_metric_value
        _str += "\ttraining_epochs:%s\n" % self.cfg.train_epochs
        _str += "\tlogging_interval:%s\n" % self.cfg.log_interval_step
        return _str

    def show_update_tensorboard_metric(self, dataset_name,metric_values,cur_epoch=0,cur_step=0,epoch_steps=0):
        ''' add metric to tensorboard

        :param dataset_name: dataset name (Train? Evaluation? Test?)
        :param metric_values: metric name and metric value
        :param cur_epoch: current epoch
        :param cur_step: current step
        :param epoch_steps : step num of one epoch
        :return:
        '''
        if dataset_name not in ['Train','Validation','Test']:
            raise RuntimeError("dataset_name (%s) must in 'Train','Validation','Test'"%dataset_name)

        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
        if self.tensorboard_writer is not None:
            for (metric_name, metric_value) in metric_values.items():
                self.tensorboard_writer.add_scalar('{}/{}_{}'.format(metric_name,dataset_name,metric_name), metric_value, cur_epoch * epoch_steps + cur_step)
            if cur_step in [0, epoch_steps - 1] or  cur_step % (self.cfg.log_interval_step * 10) == 0: # first iter, last iter and several middle iter.
                for (name, parameter) in self.model.named_parameters():
                    # if torch.isnan(parameter).int().sum() > 0 : continue
                    self.tensorboard_writer.add_histogram(name, parameter, cur_epoch * epoch_steps + cur_step)
                    if parameter.requires_grad:
                        self.tensorboard_writer.add_histogram("%s_Grad"%name, parameter.grad, cur_epoch * epoch_steps + cur_step)

        metric_str = ";\t".join("{} = {:.4f}".format(metric_name, metric_value) for (metric_name, metric_value) in metric_values.items())
        if dataset_name == 'Train':
            out_str = '[{}] {} epoch({}) step ({}/{}) {}: {}'.format(dt, "rank(%s)"% ext_dist.my_rank if ext_dist.my_rank >=0 else "",
                                                                    cur_epoch,cur_step,epoch_steps,dataset_name,metric_str)
        else:
            out_str = '[{}] {} epoch({}) {}: {}'.format(dt,"rank(%s)"%ext_dist.my_rank if ext_dist.my_rank >=0 else "",
                                                        cur_epoch,dataset_name, metric_str)
        print(out_str)
        self.logger.info(out_str)

    def _dist_wrapper(self):
        """
            wrapper model for distributed training
        """            
        if ext_dist.my_size > 1:
            self.logger.info("training with DistributedDataParallel")
            self.model = ext_dist.DDP(self.model)
            self.model.loss = MethodType(lambda obj,*args: obj.module.loss(*args) ,self.model) #  obj stands for model, both original model and transferrable model
            if self.is_transferrable: # dispatch method to model.module
                self.model.get_backbone = MethodType(lambda obj: obj.module.get_backbone() ,self.model) # obj stands for model
                self.model.get_training_metrics = MethodType(lambda obj,*args: obj.module.get_training_metrics(*args) ,self.model) #  obj stands for model

    def _is_early_stop(self, metrics_map):
        """
            check whether training achieved pre-defined metric threshold
        """
        if ext_dist.my_rank <= 0 and self.early_stopping is not None: # non distributed training, or rank 0  in distributed training
            self.early_stopping(metrics_map[self.best_metric_name], self.best_metric_value)
            return self.early_stopping.early_stop
        else:
            return False

    def train_one_epoch(self, cur_epoch, epoch_steps):
        ''' train one epoch

        :param cur_epoch: current epoch
        :param epoch_steps: how many steps of an epoch
        :return:
        '''
        self.model.train()  # set training flag
        context = self.profiler if self.profiler is not None else contextlib.nullcontext()
        with context:
            for (cur_step,(data, label)) in enumerate(self.train_dataloader):
                ##### prepare data
                '''
                Four cases of data
                Case 1 - basic: input
                Case 2 - distiller with logits: (input, (logits,seed))
                Case 3 - adapter: (input1, input2)
                Case 4 - distiller with logits and adapter: ((input1, (logits1,seed1)),(input2, (logits2,seed2))) - too complex, to be supported
                '''
                if isinstance(data, torch.Tensor):
                    # case 1
                    data = data.to(self.device)
                    label = label.to(self.device)
                elif isinstance(data, Iterable):
                    # case 2
                    data[0] = data[0].to(self.device)
                    label[0] = label[0].to(self.device)
                    # case 3
                    if len(data)>=2 and isinstance(data[1], torch.Tensor):
                        assert len(data) == len(label), "data len[%s] must equal label len[%s]"%(len(data),len(label))
                        data[1] = data[1].to(self.device)
                        label[1] = label[1].to(self.device)
                else:
                    raise RuntimeError("unknown data type:%s"%type(data))

                ##### train and update optimizer/scheduler
                self.optimizer.zero_grad()
                output = self.model(data)
                loss_value = self.model.loss(output, label)
                loss_value.backward()
                self.optimizer.step()
                if self.warmup_scheduler and cur_epoch < self.cfg.warmup_scheduler_epoch:
                    self.warmup_scheduler.step()

                ##### calculate metrics and show results
                if cur_step % self.cfg.log_interval_step == 0:
                    if self.is_transferrable: 
                        metric_values = self.model.get_training_metrics(output, label, loss_value, self.metric)
                    else:
                        metric_values = {"loss": loss_value}
                        for (metric_name, metric_fn) in sorted(self.metric.items()):
                            metric_value = metric_fn(output, label)
                            metric_value = metric_value[0] if isinstance(metric_value, list) else metric_value
                            metric_values[metric_name] = metric_value
                    self.show_update_tensorboard_metric('Train', metric_values, cur_epoch, cur_step, epoch_steps)
                
                ##### update profiler
                if context is self.profiler:
                    context.step()

    def evaluate(self, cur_epoch=0, epoch_steps=1, test_flag=True):
        ''' evaluate epoch
        :param cur_epoch : current epoch
        :param epoch_steps : steps of one epoch
        :param test_flag : whether is test or validation
        :return: metric_value_maps
        '''
        datasetName = 'Test' if test_flag else 'Validation'
        context = self.profiler if self.profiler is not None else contextlib.nullcontext()
        with torch.no_grad():
            with context:
                self.backbone.eval()  # set evaluating flag
                loss_value = 0
                metric_values = {}
                sample_num = 0
                #################### iterate on dataset ##############
                for (cur_step,(data, label)) in enumerate(self.eval_dataloader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    output = self.backbone(data)
                    output = output.logits if self.cfg.model_type.startswith("huggingface") else output
                    if isinstance(output, Iterable):
                        if not isinstance(output, torch.Tensor): # Tensor is Iterable
                            output = output[0]
                    else:
                        raise RuntimeError("Known data type:%s"%type(data))

                    batch_size = data.size(0)
                    sample_num += batch_size
                    loss_value += self.backbone.loss(output, label).item() * batch_size
                    for (metric_name,metric_fn) in self.metric.items():
                        metric_value = metric_fn(output,label)
                        metric_value = metric_value[0] if isinstance(metric_value, list) else metric_value
                        if metric_name not in metric_values:
                            metric_values[metric_name] = metric_value * batch_size
                        else:
                            metric_values[metric_name] += metric_value * batch_size
                    if cur_step % self.cfg.log_interval_step == 0:
                        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
                        print(f"{dt} {cur_step}/{epoch_steps}")
                        self.logger.info(f"{dt} {cur_step}/{epoch_steps}")
                    if context is self.profiler:
                        context.step()
                ############## average ###################
                metric_values['loss'] = loss_value
                for metric_name in sorted(metric_values.keys()):
                    metric_values[metric_name] /= sample_num
                ############## show and update tensorboard metric ###################
                self.show_update_tensorboard_metric(datasetName,metric_values,cur_epoch,cur_step=0,epoch_steps=epoch_steps)
            return metric_values

    def save_checkpoints(self, epoch=0, update_best=False, model_dir=""):
        state = {
            "epoch": epoch,
            "best_metric": self.best_metric_value,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()}
        torch.save(state, os.path.join(model_dir, "latest.pth"))
        torch.save(self.backbone.state_dict(), os.path.join(model_dir, "backbone_latest.pth"))
        if epoch % self.cfg.model_save_interval == 0:
            torch.save(state, os.path.join(model_dir, f"epoch_{epoch}.pth"))
            torch.save(self.backbone.state_dict(), os.path.join(model_dir, f"backbone_epoch_{epoch}.pth"))
        if update_best:
            torch.save(state, os.path.join(model_dir,"best.pth"))
            torch.save(self.backbone.state_dict(), os.path.join(model_dir,"backbone_best.pth"))

    def fit(self, epoch_steps, model_dir, resume=False):
        ''' train function, and save the best trained model to model_dir
        
        :param epoch_steps: steps per epoch
        :param model_dir: model saved dir
        :param resume: flag for whether resume pretrained model
        :return: validation metric. If has Earlystopping, using the best metric; Else, using the last metric
        '''
        ##### pre_process
        self._pre_process()
        very_start_time = time.time()
        initial_epoch = self.cfg.start_epoch
        train_dataset = self.train_dataloader.dataset

        ##### resume saved model
        if resume:
            latest_model_path = os.path.join(model_dir,"latest.pth")
            print(f"Resume checkpoint from {latest_model_path}")
            self.logger.info(f"Resume checkpoint from {latest_model_path}")

            state = torch.load(latest_model_path, map_location=self.device)
            initial_epoch = state["epoch"] + 1
            self.best_metric_value = state["best_metric"]
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
        
        ##### start training
        for epoch in range(initial_epoch, self.cfg.train_epochs):
            start_time = time.time()
            update_best = False
            
            ##### set epoch for dataset (used for distiller save logits function)
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)

            ##### get learning rate
            if self.scheduler is not None:
                if type(self.scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    last_lr = [item['lr'] for item in self.scheduler.optimizer.state_dict()['param_groups']]
                else:
                    last_lr = self.scheduler.get_last_lr()
                print("Epoch [%s] learning rate: %s" % (epoch, last_lr))
                self.logger.info("Epoch [%s] learning rate: %s" % (epoch, last_lr))

            ###### train and evaluate for on epoch
            self.train_one_epoch(epoch, epoch_steps)
            metrics_map = self.evaluate(epoch,epoch_steps,test_flag=False)
            
            ###### flush tensorboard and update scheduler
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.flush()
            if not (self.warmup_scheduler and epoch < self.cfg.warmup_scheduler_epoch):
                if self.cfg.lr_scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(metrics_map[self.best_metric_name])
                else:
                    self.scheduler.step()
                
            ###### update best metric value
            if metrics_map[self.best_metric_name] >= self.best_metric_value:
                self.best_metric_value = metrics_map[self.best_metric_name]
                update_best = True
                print(f"Best Epoch: {epoch}, {self.best_metric_name}: {self.best_metric_value}")
                self.logger.info(f"Best Epoch: {epoch}, {self.best_metric_name}: {self.best_metric_value}")
           
            ###### save checkpoint
            if ext_dist.my_rank <= 0: # non distributed training, or rank 0  in distributed training
                self.save_checkpoints(epoch, update_best, model_dir)

            print(f"Epoch {epoch} took {time.time()-start_time} seconds")
            self.logger.info(f"Epoch {epoch} took {time.time()-start_time} seconds")

            ###### check early stoppping
            if self._is_early_stop(metrics_map):
                self.logger.warning("Early stop after epoch:%s, the best %s is %s" % (epoch, self.best_metric_name, self.best_metric_value))
                break

        self.logger.info(F"Total time:{time.time() - very_start_time}")
        self._post_process()
        return self.best_metric_value 