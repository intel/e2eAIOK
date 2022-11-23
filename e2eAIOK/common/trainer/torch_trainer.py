import time
import utils
import random
import extend_distributed as ext_dist
from abc import ABC, abstractmethod
from data_builder import DataBuilder
from model_builder import ModelBuilder
 
class TorchTrainer(ABC):
    """
    The basic trainer class for all models

    Note:
        You should implement specfic model trainer under model folder like vit_trainer
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ext_dist.init_distributed(backend=self.cfg.dist_backend)
        self.size = ext_dist.my_size
 
    def create_model(self):
        model_builder = ModelBuilder(self.cfg)
        self.model = model_builder.create_model()

    def create_dataloader(self):
        data_builder = DataBuilder(self.cfg)
        self.data_loader = data_builder.get_data()

    def preparation(self):
        self.all_operations = utils.create_operation(self.model, self.cfg)
        
    '''
    one epoch training function, this function can be overwrite for specific model
    '''

    def train_one_epoch(self, epoch):
        self.criterion = self.all_operations['criterion']
        self.optimizer = self.all_operations['optimizer']

        # set random seed
        random.seed(epoch)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for inputs, targets in metric_logger.log_every(self.data_loader['train'], self.cfg.print_freq, header):
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
        print("Averaged stats:", metric_logger)

    '''
    evluate the validation dataset during the training, this function can be overwrite for specific model
    '''

    def evaluate(self, epoch):
        # criterion = self.all_operations['criterion']
        

        metric_logger = utils.MetricLogger(delimiter="")
        header = 'Test:'

        # switch to evaluation mode
        self.model.eval()
        
        
        for inputs, target in metric_logger.log_every(self.data_loader['val'], self.cfg.print_freq, header):
            

            output = self.model(inputs)
            
            loss = self.criterion(output, target)

            metric = utils.create_metric(output, target, self.cfg)

            batch_size = inputs.shape[0]
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
        print(output_str)

        if self.cfg.output_dir:
            checkpoint_paths = [self.cfg.output_dir + '/checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_model({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
    
    '''
    post preprocess log, metric and etc, can be overwrite
    '''
    def post_preprocess(self):
        pass

    '''
    training all epochs interface
    '''
    def fit(self):
        start_time = time.time()
        self.create_dataloader()
        self.create_model()
        self.preparation()
        if self.cfg.mode == "train":
            for i in range(1, self.cfg.train_epochs+1):
                train_start = time.time()
                self.train_one_epoch(i)
                if i % self.cfg.eval_epochs == 0:
                    eval_start = time.time()
                    self.evaluate(i)
                    print(F"Evaluate time:{time.time() - eval_start}")
                print(F"This epoch training time:{time.time() - train_start}")
            self.post_preprocess()
        else:
            eval_start = time.time()
            self.evaluate()
            print(F"Evaluate time:{time.time() - eval_start}")
        
                
        print(F"Total time:{time.time() - start_time}")