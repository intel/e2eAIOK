import argparse
import subprocess
import yaml
from multiprocessing import cpu_count

from Launcher.BaseModelLauncher import BaseModelLauncher

class WnDLauncher(BaseModelLauncher):
    '''
        Wide and Deep sigopt model optimization launcher
    '''
    def __init__(self, dataset_format, dataset_meta_path, train_path, eval_path, args):
        super().__init__(dataset_format, dataset_meta_path, train_path, eval_path, args)
        args = self.parse_args(args)

        if dataset_format == 'TFRecords':
            self.params['prebatch_size'] = args.prebatch_size
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
        cores = self.params['cores'] if 'cores' in self.params else cpu_count()
        ppn = self.params['ppn'] if 'ppn' in self.params else 1
        ccl_worker_num = self.params['ccl_worker_num']
        hosts = self.params['hosts']
        python_executable = self.params['python_executable']
        omp_threads = cores // 2 // ppn - ccl_worker_num
        ranks = len(hosts) * ppn

        # construct WnD launch command with mpi
        cmd = f"time mpirun -genv OMP_NUM_THREADS={omp_threads} -map-by socket -n {ranks} -ppn {ppn} -hosts {','.join(hosts)} -print-rank-map "
        cmd += f"-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 "
        cmd += f"{python_executable} models/WnD/sigopt_runner.py "
        cmd += f"--dataset_format {self.params['dataset_format']} " \
            + f"--train_data_pattern {self.params['train_dataset_path']} --eval_data_pattern {self.params['eval_dataset_path']} --transformed_metadata_path {self.params['dataset_meta_path']} " \
            + f"--global_batch_size {self.params['global_batch_size']} --eval_batch_size {self.params['global_batch_size']} --num_epochs {self.params['num_epochs']} " \
            + f"--metric {self.params['metric']} --metric_threshold {self.params['metric_threshold']} "
        if self.params['dataset_format'] == 'TFRecords':
            cmd += f"--prebatch_size {self.params['prebatch_size']} "
        if self.params['linear_learning_rate'] != -1:
            cmd += f"--linear_learning_rate {self.params['linear_learning_rate']} "
        if self.params['deep_learning_rate'] != -1:
            cmd += f"--deep_learning_rate {self.params['deep_learning_rate']} "
        if self.params['deep_warmup_epochs'] != -1:
            cmd += f"--deep_warmup_epochs {self.params['deep_warmup_epochs']} "
        if len(self.params['deep_hidden_units']) != 0:
            cmd += f"--deep_hidden_units {' '.join([str(item) for item in self.params['deep_hidden_units']])} "
        if self.params['deep_dropout'] != -1:
            cmd += f"--deep_dropout {self.params['deep_dropout']} "
        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()