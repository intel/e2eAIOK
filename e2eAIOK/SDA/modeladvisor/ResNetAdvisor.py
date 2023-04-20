import subprocess #nosec
import yaml
import logging
import time
import sys

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor


class ResNetAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')
        self.train_script = "/home/vmagent/app/e2eaiok/modelzoo/resnet/mlperf_resnet/imagenet_main.py"
        self.python_path = "/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/"
        self.train_python = f"{self.python_path}/python"
        self.horovodrun_path = f"{self.python_path}/horovodrun"


    def update_metrics(self):
        metrics = []
        metrics.append({'name': 'accuracy', 'value': self.mean_accuracy})
        metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments=None):
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters['label_smoothing'] = assignments['label_smoothing']
            tuned_parameters['resnet_size'] = assignments['resnet_size']
            tuned_parameters['num_filters'] = assignments['num_filters']
            tuned_parameters['kernel_size'] = assignments['kernel_size']
            tuned_parameters['momentum'] = assignments['momentum']
            tuned_parameters['weight_decay'] = assignments['weight_decay']
            tuned_parameters['base_lr'] = assignments['base_lr']
        else:
            tuned_parameters['label_smoothing'] = 0.0
            tuned_parameters['resnet_size'] = 50
            tuned_parameters['num_filters'] = 64
            tuned_parameters['kernel_size'] = 7
            tuned_parameters['momentum'] = 0.9
            tuned_parameters['weight_decay'] = 1e-4
            tuned_parameters['base_lr'] = 0.128
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='resnet_sigopt.yaml'):
        config = {}
        config['project'] = self.params['project']
        config['experiment'] = self.params['experiment']
        parameters = [{'name': 'label_smoothing', 'bounds': {'min': 0.05, 'max': 0.5}, 'type': 'double'},
                      {'name': 'resnet_size', 'grid': [18, 34, 50], 'type': 'int'},
                      {'name': 'num_filters', 'grid': [2,4,8,16,32,64], 'type': 'int'},
                      {'name': 'kernel_size', 'grid': [1,3,5,7,9], 'type': 'int'},
                      {'name': 'momentum', 'bounds': {'min': 0.6, 'max': 0.99}, 'type': 'double'},
                      {'name': 'weight_decay', 'bounds': {'min': 1e-5, 'max': 5e-5}, 'type': 'double'},
                      {'name': 'base_lr', 'bounds': {'min': 0.01, 'max': 0.2}, 'type': 'double'},
                      ]
                      
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
        return config

    def train_model(self, args):
        start_time = time.time()
        if args['ppn'] == 1 and len(args['hosts']) == 1:
            self.dist_launch(args)
        else:
            self.dist_launch(args)
        self.training_time = time.time() - start_time
        with open("/home/vmagent/app/e2eaiok/modelzoo/resnet/mlperf_resnet/metric.txt",'r') as f:
            lines = f.readlines()
        self.mean_accuracy = float(lines[-1])
        metrics = self.update_metrics()
        model_path = self.model_saved_path
        return self.training_time, model_path, metrics

    def launch(self, args):
        # construct ResNet launch command
        self.model_saved_path = args['model_saved_path']
        cmd = f"{self.train_python} -u {self.train_script} '123456' "
        cmd += f"--label_smoothing '{args['model_parameter']['tuned_parameters']['label_smoothing']}' --num_filters '{args['model_parameter']['tuned_parameters']['num_filters']}' --data_dir '{args['train_dataset_path']}' " \
            + f"--model_dir '{self.model_saved_path}' --train_epochs '{args['num_epochs']}' " \
            + f"--stop_threshold '{args['metric_threshold']}' --batch_size '{args['global_batch_size']}' --version 1 --resnet_size '{args['model_parameter']['tuned_parameters']['resnet_size']}' " \
            + f"--epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 " \
            + f"--enable_lars --weight_decay '{args['model_parameter']['tuned_parameters']['weight_decay']}' --kernel_size '{args['model_parameter']['tuned_parameters']['kernel_size']}'"
        if args["use_synthetic_data"]:
            cmd += f"--use_synthetic_data"
        
        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait() 

    def dist_launch(self, args):
        # construct ResNet distributed launch command
        ppn = args['ppn']
        hosts = ",".join(str(i)+":"+str(ppn) for i in args['hosts']) 
        self.model_saved_path = args['model_saved_path']
        ranks = len(args['hosts']) * ppn
        cmd = f"{self.horovodrun_path} -n {ranks} -H {hosts} HOROVOD_CPU_OPERATIONS=CCL CCL_ATL_TRANSPORT=mpi "
        cmd += f"{self.train_python} -u {self.train_script} '123456' "
        cmd += f"--label_smoothing '{args['model_parameter']['tuned_parameters']['label_smoothing']}' --num_filters '{args['model_parameter']['tuned_parameters']['num_filters']}' --data_dir '{args['train_dataset_path']}' " \
            + f"--model_dir '{self.model_saved_path}' --train_epochs '{args['num_epochs']}' " \
            + f"--stop_threshold '{args['metric_threshold']}' --batch_size '{204}' --version 1 --resnet_size '{args['model_parameter']['tuned_parameters']['resnet_size']}' " \
            + f"--epochs_between_evals 1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 " \
            + f"--enable_lars --weight_decay '{args['model_parameter']['tuned_parameters']['weight_decay']}' --kernel_size '{args['model_parameter']['tuned_parameters']['kernel_size']}' " \
            + f"--base_lr '{args['model_parameter']['tuned_parameters']['base_lr']}' --momentum '{args['model_parameter']['tuned_parameters']['momentum']}' "

        if args["use_synthetic_data"]:
            cmd += f"--use_synthetic_data"

        self.logger.info(f'training launch command: {cmd}')
        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait() 
