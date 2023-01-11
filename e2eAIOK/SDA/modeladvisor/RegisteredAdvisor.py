import subprocess
import yaml
import logging
import time

from e2eAIOK.common.utils import *
from e2eAIOK.utils.hydroconfig import default_settings
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class RegisteredAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path=None, train_path=None, eval_path=None, settings={'model_name':'custom_registered'}):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

        self.experiment_name = "sklearn"
        self.metrics = [("mean_accuracy", "maximize")]
        self.training_time_as_metrics = False
        self.parameters = {
            'max_depth':11,
            'learning_rate':float(0.9),
            'min_split_loss':float(7)
        }
        self.sigopt_config = [{
            'name': 'learning_rate',
            'bounds': {
                'min': 0.0,
                'max': 1.0
            },
            'type': 'double'
        }]
        self.execute_cmd_base = "/opt/intel/oneapi/intelpython/latest/bin/python /home/vmagent/app/e2eaiok/example/sklearn_train.py"
        self.result_file_name = "result"
        self.observation_budget = 1
        self.sigopt_enable_parameters = []
        self.sigopt_list_parameters = {}

    def __fix(self, sigopt_config):
        for k, v in enumerate(sigopt_config):
            if 'grid' in v:
                if ('type' in v and v['type'] in ['str', 'bool']) or (not 'type' in v):
                     # handle config with string grid
                     # handle config with bool config
                     max_len = len(v['grid'])
                     self.sigopt_list_parameters[v['name']] = v['grid']
                     sigopt_config[k]['grid'] = list(range(max_len))
                     sigopt_config[k]['type'] = 'int'
            if len(v) == 1 and 'name' in v:
                # handle config with no option
                sigopt_config[k]['grid'] = [0, 1]
                sigopt_config[k]['type'] = 'int'
                self.sigopt_enable_parameters.append(v['name'])

        return sigopt_config


    # ====== Implementation of required methods ======

    def update_metrics(self):
        metrics = []
        for metric in self.metrics:
            metrics.append({'name': metric[0], 'value': self.result})
        if self.training_time_as_metrics:
            metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments=None):
        config = {}
        tuned_parameters = {}
        if assignments:
            for k in self.parameters.keys():
                #TODO: need to consider scenarios of grid and enabled/disable
                tuned_parameters[k] = assignments[k]
        else:
            tuned_parameters = self.parameters
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='test_sigopt.yaml'):
        config = {}
        config['project'] = 'e2eaiok'
        config['experiment'] = self.experiment_name
        config['parameters'] = self.__fix(self.sigopt_config)
        config['metrics'] = []
        for metric in self.metrics:
            config['metrics'].append({
            'name': metric[0],
            'strategy': 'optimize',
            'objective': metric[1]
        })
        if self.training_time_as_metrics:
            config['metrics'].append({
                'name': 'training_time',
                'objective': 'minimize'
            })
        config['observation_budget'] = self.observation_budget

        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def train_model(self, args):
        start_time = time.time()
        self.result, model_path = self.dist_launch(args)
        self.training_time = time.time() - start_time
        metrics = self.update_metrics()
        return self.training_time, model_path, metrics

    def dist_launch(self, args):
        # construct WnD launch command with mpi
        model_saved_path = args['model_saved_path']
        cmd = []
        cmd += self.execute_cmd_base.split()
        for key in self.parameters.keys():
            if key in self.sigopt_enable_parameters:
                if args['model_parameter']['tuned_parameters'][key] == 0:
                    continue
                else:
                    args['model_parameter']['tuned_parameters'][key] = None
            if key in self.sigopt_list_parameters:
                args['model_parameter']['tuned_parameters'][key] = self.sigopt_list_parameters[key][args['model_parameter']['tuned_parameters'][key]]
            cmd.append(f"--{key}")
            if args['model_parameter']['tuned_parameters'][key]:
                cmd.append(f"{args['model_parameter']['tuned_parameters'][key]}")

        cmd.append(f"--saved_path")
        cmd.append(f"{model_saved_path}")
        
        self.logger.info(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=False)
        process.wait()

        # parse mrr
        with open(os.path.join(model_saved_path, self.result_file_name), "r") as f:
            result = float(f.read())
        return result, model_saved_path
