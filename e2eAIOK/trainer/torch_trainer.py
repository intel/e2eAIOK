import time
import utils
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
    
    @abstractmethod
    def create_model(self):
        model_builder = ModelBuilder(self.cfg)
        self.model = model_builder.create_model()

    @abstractmethod
    def create_dataloader(self):
        data_builder = DataBuilder(self.cfg)
        self.data_loader = data_builder.get_data()

    @abstractmethod
    def preparation(self):
        self.all_operations = create_operation(self.model, self.cfg)
        
    '''
    one epoch training function, this function can be overwrite for specific model
    '''
    @abstractmethod
    def train_one_epoch(self, epoch):
        criterion = self.all_operations['criterion']
        optimizer = self.all_operations['optimizer']

        # set random seed
        random.seed(epoch)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)


        for inputs, targets in metric_logger.log_every(data_loader, print_freq=self.cfg.print_freq, header):
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            loss_value = loss.item()

            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()
        
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    '''
    evluate the validation dataset during the training, this function can be overwrite for specific model
    '''
    @abstractmethod
    def evaluate(self):
        criterion = self.all_operations['critetion']
        metric = self.all_operations['metric']

        metric_logger = utils.MetricLogger(delimiter="")
        header = 'Test:'

        # switch to evaluation mode
        model.eval()
        
        
        for inputs, target in metric_logger.log_every(data_loader, print_freq=self.cfg.print_freq, header):
            

            output = model(inputs)
            
            loss = criterion(output, target)

            metric = create_metric(output, target, cfg)

            batch_size = inputs.shape[0]
            metric_logger.update(loss=loss.item())
            for i in range(len(metric)):
                metric_logger.meters[metric.keys()[i]].update(metric.values()[i], n=self.cfg.batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                "{}: {}".format(name, str(meter))
            )
        print(output_str)

        if self.cfg.output_dir:
            checkpoint_paths = [self.cfg.output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_model({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
    
    '''
    post preprocess log, metric and etc, can be overwrite
    '''
    @abstractmethod
    def post_preprocess(self):
        pass

    '''
    training all epochs interface
    '''
    @abstractmethod
    def fit(self):
        start_time = time.time()
        self.create_dataloader()
        self.create_model()
        self.preparation()
        if self.cfg.mode == "train":
            for i in range(self.cfg.train_epochs):
                train_start = time.time()
                self.train_one_epoch(epoch)
                if i % self.cfg.eval_epochs == 0:
                    eval_start = time.time()
                    self.evaluate()
                    print(F"Evaluate time:{time.time() - eval_start}")
                print(F"This epoch training time:{time.time() - train_start}")
            self.post_preprocess()
        else:
            eval_start = time.time()
            self.evaluate()
            print(F"Evaluate time:{time.time() - eval_start}")
        
                
        print(F"Total time:{time.time() - start_time}")