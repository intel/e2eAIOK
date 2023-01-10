import time
from e2eAIOK.common.trainer.utils import utils
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist

class CVTrainer(TorchTrainer):
    def __init__(self, cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric):
        super().__init__(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        self.best_acc = 0
    def _dist_wrapper(self):
        """
            wrapper model for distributed training
        """
        if ext_dist.my_size > 1:
            self.model = ext_dist.DDP(self.model, find_unused_parameters=True)
    def _is_early_stop(self, metric):
        """
            check whether training achieved pre-defined metric threshold
        """
        return self.best_acc >= self.cfg["metric_threshold"]

    def train_one_epoch(self, epoch):
        """
            train one epoch
        """
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        self.model.train()

        for inputs, targets in metric_logger.log_every(self.train_dataloader, self.cfg.print_freq, header):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss_value = loss.item()
            self.optimizer.zero_grad()       
            loss.backward()
            self.optimizer.step()
        
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

    def evaluate(self, epoch):
        """
            evaluatiuon
        """
        metric_logger = utils.MetricLogger(delimiter="")

        header = 'Test:'

        self.model.eval()
        
        for inputs, target in metric_logger.log_every(self.eval_dataloader, self.cfg.print_freq, header):
            output = self.model(inputs)
            loss = self.criterion(output, target)
            metric = self.metric(output, target)
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(metric[0].item(), n=self.cfg.eval_batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, losses=metric_logger.loss))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        
        if metric_logger.acc1.global_avg >= self.best_acc:
            self.best_acc = metric_logger.acc1.global_avg

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
        for i in range(1, self.cfg.train_epochs+1):
            train_start = time.time()
            self.train_one_epoch(i)
            self.scheduler.step(i)
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
        return self.best_acc