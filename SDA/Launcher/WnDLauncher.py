import argparse
import subprocess
import yaml
from multiprocessing import cpu_count

from Launcher.BaseModelLauncher import BaseModelLauncher

class WnDLauncher(BaseModelLauncher):
    '''
        Wide and Deep sigopt model optimization launcher
    '''
    def __init__(self, dataset_meta_path, train_path, eval_path, args):
        super().__init__(dataset_meta_path, train_path, eval_path, args)
        args = self.parse_args(self.cmdl_args)

        # model params
        self.params['deep_hidden_units'] = args.deep_hidden_units
        self.params['deep_dropout'] = args.deep_dropout
        # train params
        self.params['linear_learning_rate'] = args.linear_learning_rate
        self.params['deep_learning_rate'] = args.deep_learning_rate
        self.params['deep_warmup_epochs'] = args.deep_warmup_epochs

        self.generate_sigopt_yaml()

    def parse_args(self, args):
        '''
            Add model specific parameters
            For sigopt parameters, set parameter explicitly and the parameter will not be optimized by sigopt
        '''
        self.parser.add_argument('--prebatch_size', type=int, default=4096, help='Dataset prebatch size, only applyable for TFRecords format')
        self.parser.add_argument('--deep_hidden_units', type=int, default=[], nargs='+', help='Hidden units per layer for deep model, separated by spaces')
        self.parser.add_argument('--linear_learning_rate', type=float, default=-1, help='Learning rate for linear model')
        self.parser.add_argument('--deep_learning_rate', type=float, default=-1, help='Learning rate for deep model')
        self.parser.add_argument('--deep_warmup_epochs', type=float, default=-1, help='Number of learning rate warmup epochs for deep model')
        self.parser.add_argument('--deep_dropout', type=float, default=-1, help='Dropout regularization for deep model')
        return self.parser.parse_args(args)
    
    def generate_sigopt_yaml(self, file='models/WnD/sigopt.yaml'):
        config = {}
        config['project'] = 'sda'
        config['experiment'] = 'WnD'
        parameters = [{'name': 'dnn_hidden_unit1', 'grid': [64, 128, 256, 512, 1024, 2048], 'type': 'int'}, 
                      {'name': 'dnn_hidden_unit2', 'grid': [64, 128, 256, 512, 1024, 2048], 'type': 'int'}, 
                      {'name': 'dnn_hidden_unit3', 'grid': [64, 128, 256, 512, 1024, 2048], 'type': 'int'}, 
                      {'name': 'deep_learning_rate', 'bounds': {'min': 1.0e-4, 'max': 1.0e-1}, 'type': 'double', 'transformation': 'log'},
                      {'name': 'linear_learning_rate', 'bounds': {'min': 1.0e-2, 'max': 1.0}, 'type': 'double', 'transformation': 'log'}, 
                      {'name': 'deep_warmup_epochs', 'bounds': {'min': 1, 'max': 8}, 'type': 'int'},
                      {'name': 'deep_dropout', 'bounds': {'min': 0, 'max': 0.5}, 'type': 'double'}]
        config['parameters'] = parameters
        metrics = []
        if 'training_time_threshold' in self.params:
            metrics.append({'name': 'training_time', 'objective': 'minimize', 'threshold': self.params['training_time_threshold']})
            metrics.append({'name': self.params['metric'], 'objective': self.params['metric_objective'], 'threshold': self.params['metric_threshold']})
        else:
            metrics.append({'name': self.params['metric'], 'objective': self.params['metric_objective']})
        config['metrics'] = metrics
        config['observation_budget'] = self.params['observation_budget']
        with open(file, 'w') as f:
            yaml.dump(config, f)
    
    def launch(self):
        cmd = f"{self.params['python_executable']} models/WnD/sigopt_runner.py " \
            + f"--dataset_meta_path {self.params['dataset_meta_path']} " \
            + f"--train_dataset_path '{self.params['train_dataset_path']}' " \
            + f"--eval_dataset_path '{self.params['eval_dataset_path']}' " \
            + f"{' '.join(self.cmdl_args)}"
        print(f'sigopt runner launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()