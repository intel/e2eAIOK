import subprocess
import yaml
import logging
import time
import os

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class DLRMAdvisor(BaseModelAdvisor):
    '''
        Wide and Deep sigopt model optimization launcher
    '''
    def __init__(self, dataset_meta_path, train_path, eval_path, args):
        args = setting_local_default(args)
        super().__init__(dataset_meta_path, train_path, eval_path, args)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')
        self.train_python = "/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python"
        self.train_script = "/home/vmagent/app/e2eaiok/modelzoo/dlrm/dlrm/launch.py"
        self.train_path = train_path
        self.test_path = eval_path
        self.dataset_meta_path = dataset_meta_path

    def initialize_model_parameter(self, assignments=None):
        '''
            Add model specific parameters
            For sigopt parameters, set parameter explicitly and the parameter will not be optimized by sigopt
        '''
        config = {}
        tuned_parameters = {}
        if assignments:
            mlp_top_size = ["1024-1024-512-256-1","512-512-256-128-1","512-256-128-1","512-256-1","256-128-1","128-64-1","256-1","128-1"]
            mlp_bot_size = ["13-512-256-","13-512-256-","13-256-","13-128-"]
            tuned_parameters['mlp_top_size'] = assignments["mlp_top_size"]
            tuned_parameters['mlp_bot_size'] = assignments["mlp_bot_size"]
            tuned_parameters['lamb_lr'] = str(assignments['lamb_lr'])
            tuned_parameters['learning_rate'] = str(assignments['learning_rate'])
            tuned_parameters['lr_num_warmup_steps'] = str(assignments['lr_num_warmup_steps'])
            tuned_parameters['lr_decay_start_step'] = str(assignments['lr_decay_start_step'])
            tuned_parameters['lr_num_decay_steps'] = str(assignments['lr_num_decay_steps'])
            tuned_parameters['arch_sparse_feature_size'] = str(assignments["arch_sparse_feature_size"])
            tuned_parameters['arch_mlp_top'] = str(mlp_top_size[tuned_parameters["mlp_top_size"]])
            tuned_parameters['arch_mlp_bot'] = str(mlp_bot_size[tuned_parameters["mlp_bot_size"]])+str(assignments["arch_sparse_feature_size"])
        else:
            mlp_top_size = ["1024-1024-512-256-1","512-512-256-128-1","512-256-128-1","512-256-1","256-128-1","128-64-1","256-1","128-1"]
            mlp_bot_size = ["13-512-256-","13-512-256-","13-256-","13-128-"]
            tuned_parameters['mlp_top_size'] = 4
            tuned_parameters['mlp_bot_size'] = 3
            tuned_parameters['lamb_lr'] = "16"
            tuned_parameters['learning_rate'] = "16"
            tuned_parameters['lr_num_warmup_steps'] = "4000"
            tuned_parameters['lr_decay_start_step'] = "5760"
            tuned_parameters['lr_num_decay_steps'] = "17000"
            tuned_parameters['arch_sparse_feature_size'] = 128
            tuned_parameters['arch_mlp_top'] = str(mlp_top_size[tuned_parameters['mlp_top_size']])
            tuned_parameters['arch_mlp_bot'] = str(mlp_bot_size[tuned_parameters['mlp_bot_size']])+str(128)
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config
    
    ###### Implementation of required methods ######

    def generate_sigopt_yaml(self, file='dlrm_sigopt.yaml'):
        config = {}
        config['project'] = 'dlrm'
        config['experiment'] = 'DLRM'
        parameters = [{'name':'learning_rate','bounds':{'min':5,'max':50},'type':'int'},
                      {'name':'lamb_lr','bounds':{'min':5,'max':50},'type':'int'},
                      {'name':'lr_num_warmup_steps','bounds':{'min':2000,'max':4500},'type':'int'},
                      {'name':'lr_decay_start_step','bounds':{'min':4501,'max':9000},'type':'int'},
                      {'name':'lr_num_decay_steps','bounds':{'min':5000,'max':15000},'type':'int'},
                      {'name':'arch_sparse_feature_size','grid': [128,64,16],'type':'int'},
                      {'name':'mlp_top_size','bounds':{'min':0,'max':7},'type':'int'},
                      {'name':'mlp_bot_size','bounds':{'min':0,'max':3},'type':'int'},]
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

    def update_metrics(self):
        metrics = []
        metrics.append({'name': 'accuracy', 'value': self.mean_accuracy})
        metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']
  
    def train_model(self, args):
        start_time = time.time()
        if args['ppn'] == 1 and len(args['hosts']) == 1:
            self.launch(args)
        else:
            self.dist_launch(args)
        self.training_time = time.time() - start_time
        with open("./best_auc.txt",'r') as f:
            lines = f.readlines()
        self.mean_accuracy = float(lines[-1])
        metrics = self.update_metrics()
        model_path = args['model_saved_path']
        return self.training_time, model_path, metrics

    def launch(self, args):
        # construct WnD launch command
        cmd = f"{self.train_python} -u {self.train_script} "
        model_saved_path = args['model_saved_path']
        cmd +=f"/home/vmagent/app/e2eaiok/modelzoo/dlrm/dlrm/dlrm_s_pytorch.py --mini-batch-size={args['train_batch_size']} --print-freq=16  " \
            + f"--test-mini-batch-size={args['test_batch_size']} --test-freq=800 " \
            + f"--train-data-path={self.train_path} --eval-data-path={self.test_path} " \
            + f"--nepochs=1 --day-feature-count={args['data_path'] + '/day_fea_count.npz'} " \
            + f"--loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex " \
            + f" --dist-backend=ccl --print-time --data-generation=dataset --optimizer=1 --bf16 --data-set=terabyte " \
            + f"--sparse-dense-boundary=403346  --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 " \
            + f"--mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=12345 " \
            + f"--arch-sparse-feature-size={args['model_parameter']['tuned_parameters']['arch_sparse_feature_size']} " \
            + f"--arch-mlp-bot={args['model_parameter']['tuned_parameters']['arch_mlp_bot']} " \
            + f"--arch-mlp-top={args['model_parameter']['tuned_parameters']['arch_mlp_top']} " \
            + f"--lamblr={args['model_parameter']['tuned_parameters']['lamb_lr']} --save-model={model_saved_path} " \
            + f"--learning-rate={args['model_parameter']['tuned_parameters']['learning_rate']} " \
            + f"--lr-num-warmup-steps={args['model_parameter']['tuned_parameters']['lr_num_warmup_steps']} " \
            + f"--lr-decay-start-step={args['model_parameter']['tuned_parameters']['lr_decay_start_step']} " \
            + f"--lr-num-decay-steps={args['model_parameter']['tuned_parameters']['lr_num_decay_steps']}"
        self.logger.info(f'### Starting model training ###, launch cmd is: \n{cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
    
    def dist_launch(self, args):
        ppn = args['ppn']
        hosts = args['hosts']
        model_saved_path = args['model_saved_path']
        if len(args['hosts']) == 1:
            cmd = f"{self.train_python} -u {self.train_script} --distributed --nproc_per_node={ppn} --nnodes={len(hosts)} --hostfile {','.join(hosts)} "
        else:
            hostfile = args['hostfile']
            cmd = f"{self.train_python} -u {self.train_script}  --distributed --nproc_per_node={ppn} --nnodes={len(hosts)} --hostfile {hostfile} "  
        cmd +=f"/home/vmagent/app/e2eaiok/modelzoo/dlrm/dlrm/dlrm_s_pytorch.py --mini-batch-size={args['train_batch_size']} --print-freq=16  " \
            + f"--test-mini-batch-size={args['test_batch_size']} --test-freq=800 " \
            + f"--train-data-path={self.train_path} --eval-data-path={self.test_path} " \
            + f"--nepochs=1 --day-feature-count={args['data_path'] + '/day_fea_count.npz'} " \
            + f"--loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex " \
            + f" --dist-backend=ccl --print-time --data-generation=dataset --optimizer=1 --bf16 --data-set=terabyte " \
            + f"--sparse-dense-boundary=403346  --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 " \
            + f"--mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=12345 " \
            + f"--arch-sparse-feature-size={args['model_parameter']['tuned_parameters']['arch_sparse_feature_size']} " \
            + f"--arch-mlp-bot={args['model_parameter']['tuned_parameters']['arch_mlp_bot']} " \
            + f"--arch-mlp-top={args['model_parameter']['tuned_parameters']['arch_mlp_top']} " \
            + f"--lamblr={args['model_parameter']['tuned_parameters']['lamb_lr']} --save-model={model_saved_path} " \
            + f"--learning-rate={args['model_parameter']['tuned_parameters']['learning_rate']} " \
            + f"--lr-num-warmup-steps={args['model_parameter']['tuned_parameters']['lr_num_warmup_steps']} " \
            + f"--lr-decay-start-step={args['model_parameter']['tuned_parameters']['lr_decay_start_step']} " \
            + f"--lr-num-decay-steps={args['model_parameter']['tuned_parameters']['lr_num_decay_steps']}"
        self.logger.info(f'### Starting model training ###, launch cmd is: \n{cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait() 

def setting_local_default(settings):
    default = {
        "ppn": 2,
        "ccl_worker_num": 4,
        "train_batch_size": 262144,
        "test_batch_size": 131072,
        "num_epochs": 1,
        "cores": None,
        "training_time_threshold": 10000,
        "metric": "accuracy",
        "metric_objective": "maximize",
        "metric_threshold": 0.6,
    }
    for k, v in default.items():
        if k not in settings:
            settings[k] = v

    return settings
