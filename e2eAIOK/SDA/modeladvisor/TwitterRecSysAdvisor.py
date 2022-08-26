import subprocess
import yaml
import logging
import time

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class TwitterRecSysAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

        self.train_path = train_path
        self.test_path = eval_path
        self.dataset_meta_path = dataset_meta_path
        self.saved_path = self.params['save_path']
        self.train_python = "/opt/intel/oneapi/intelpython/latest/bin/python"
        self.train_script = "/home/vmagent/app/e2eaiok/modelzoo/TwitterRecSys2021/model_e2eaiok/xgboost/train.py"

    def update_metrics(self):
        result_metrics_path = os.path.join(self.params['model_saved_path'],
                                           "result.yaml")
        if not os.path.exists(result_metrics_path):
            raise FileNotFoundError(
                f"{self.train_script} completed, while we can't find \
                    result {result_metrics_path} file.")
        with open(result_metrics_path) as f:
            results = yaml.load(f, Loader=yaml.FullLoader)
        metrics = []
        metrics.append({'name': 'AP', 'value': results['AP']})
        metrics.append({'name': 'RCE', 'value': results['RCE']})

        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments = None):
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters['max_depth'] = assignments['max_depth']
            tuned_parameters['learning_rate'] = assignments['learning_rate']
            tuned_parameters['subsample'] = assignments['subsample']
            tuned_parameters['colsample_bytree'] = assignments['colsample_bytree']
            tuned_parameters['num_boost_round'] = assignments['num_boost_round']
        else:
            tuned_parameters['max_depth'] = 8
            tuned_parameters['learning_rate'] = float(0.1)
            tuned_parameters['subsample'] = float(0.8)
            tuned_parameters['colsample_bytree'] = float(0.8)
            tuned_parameters['num_boost_round'] = 250

        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='RecSys_sigopt.yaml'):
        config = {}
        config['project'] = 'e2eaiok'
        config['experiment'] = 'RecSys'
        parameters = [{'name': 'max_depth', 'bounds': {'min': 5, 'max': 20}, 'type': 'int'},
                    {'name': 'learning_rate', 'bounds': {'min': 0.0, 'max': 1.0}, 'type': 'double'},
                    {'name': 'subsample', 'bounds': {'min': 0.5, 'max': 1}, 'type': 'double'},
                    {'name': 'colsample_bytree', 'bounds': {'min': 0.5, 'max': 1.0}, 'type': 'double'},
                    {'name': 'num_boost_round', 'bounds': {'min': 100, 'max': 1000}, 'type': 'int'}]
        user_defined_parameter = self.params['model_parameter']['parameters'] if ('model_parameter' in self.params) and ('parameters' in self.params['model_parameter']) else None
        config['parameters'] = parameters
        if user_defined_parameter:
            self.logger.info(f"Update with user defined parameters {user_defined_parameter}")
            update_list(config['parameters'], user_defined_parameter)
        
        config['metrics'] = [
            {'name': 'AP', 'strategy': 'optimize', 'objective': 'maximize'},
            {'name': 'RCE', 'objective': 'maximize'}
        ]
        user_defined_metrics = self.params['model_parameter']['metrics'] if ('model_parameter' in self.params) and ('metrics' in self.params['model_parameter']) else None
        if user_defined_metrics:
            self.logger.info(f"Update with user defined parameters {user_defined_metrics}")
            update_list(config['metrics'], user_defined_metrics)
        config['observation_budget'] = self.params['observation_budget']

        # save to local disk
        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def train_model(self, args):
        start_time = time.time()
        self.launch(args)
        self.training_time = time.time() - start_time
        metrics = self.update_metrics()
        return self.training_time, args['model_saved_path'], metrics
    
    def launch(self, args):
        cmd = f"{self.train_python} -u "
        cmd += f"{self.train_script} "
        cmd += f"--stage {args['stage']} "
        cmd += f"--target {args['target']} "
        cmd += f"--train_data_path {self.train_path} "
        cmd += f"--valid_data_path {self.test_path} "
        cmd += f"--model_save_path {args['model_saved_path']} "
        cmd += f"--max_depth {args['model_parameter']['tuned_parameters']['max_depth']} "
        cmd += f"--learning_rate {args['model_parameter']['tuned_parameters']['learning_rate']} "
        cmd += f"--subsample {args['model_parameter']['tuned_parameters']['subsample']} "
        cmd += f"--colsample_bytree {args['model_parameter']['tuned_parameters']['colsample_bytree']} "
        cmd += f"--num_boost_round {args['model_parameter']['tuned_parameters']['num_boost_round']} "
        self.logger.info(f'training launch command: {cmd}')

        process = subprocess.Popen(cmd, shell=True)
        process.wait()

