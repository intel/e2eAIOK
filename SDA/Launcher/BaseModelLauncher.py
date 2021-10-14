import argparse
from abc import abstractmethod

class BaseModelLauncher:
    def __init__(self, dataset_format, dataset_meta_path, train_path, eval_path, model_args):
        self.cmdl_args = model_args
        self.parser = argparse.ArgumentParser()
        self.params = {}
        self.params['dataset_format'] = dataset_format
        self.params['dataset_meta_path'] = dataset_meta_path
        self.params['train_dataset_path'] = train_path
        self.params['eval_dataset_path'] = eval_path

        args, _ = self.parse_common_args(model_args)
        self.params['ppn'] = args.ppn
        if args.cores is not None:
            self.params['cores'] = args.cores
        self.params['hosts'] = args.hosts
        self.params['ccl_worker_num'] = args.ccl_worker_num
        self.params['python_executable'] = args.python_executable
        self.params['global_batch_size'] = args.global_batch_size
        self.params['num_epochs'] = args.num_epochs
        if dataset_format == 'TFRecords':
            self.params['trainset_size'] = args.trainset_size
        self.params['model_dir'] = args.model_dir
        self.params['metric'] = args.metric
        self.params['metric_threshold'] = args.metric_threshold
        self.params['metric_objective'] = args.metric_objective
        self.params['observation_budget'] = args.observation_budget
        if args.training_time_threshold is not None:
            self.params['training_time_threshold'] = args.training_time_threshold

    def parse_common_args(self, args):
        self.parser.add_argument('--ppn', type=int, default=1, help='Define worker number per node for distributed training')
        self.parser.add_argument('--cores', type=int, help='Define node CPU cores used for training')
        self.parser.add_argument('--hosts', type=str, default=['localhost'], nargs='+', help='Hosts to launch training, separated by spaces')
        self.parser.add_argument('--ccl_worker_num', type=int, default=1, help='CCL woker number')
        self.parser.add_argument('--python_executable', type=str, default='python', help='Python interpreter')
        self.parser.add_argument('--program', type=str, help='The path to the program to be launched')

        self.parser.add_argument('--global_batch_size', type=int, default=1024, help='Global batch size for train and evaluation')
        self.parser.add_argument('--num_epochs', type=int, default=1, help='Number training epochs')
        self.parser.add_argument('--trainset_size', type=int, help='Define trainset size')
        self.parser.add_argument('--model_dir', type=str, default='./', help='Model save path')
        self.parser.add_argument('--metric', type=str, default='AUC', help='Model evaluation metric')
        self.parser.add_argument('--observation_budget', type=int, default=40, help='Define total number of sigopt optimization loop')
        self.parser.add_argument('--metric_threshold', type=float, default=0, help='Model evaluation metric threshold')
        self.parser.add_argument('--metric_objective', type=str, default='maximize', help='Either set minimize or maximize')
        self.parser.add_argument('--training_time_threshold', type=int, help='Define training time threshold for sigopt optmization metric')
        return self.parser.parse_known_args(args)
    
    def parse_args(self, args):
        raise NotImplementedError('Parse args for model specific args is not supported in BaseModelLauncher')

    @abstractmethod
    def generate_sigopt_yaml():
        pass

    @abstractmethod
    def launch(self):
        pass