import subprocess
import yaml
import logging
import time


from e2eAIOK.common.utils import *
from e2eAIOK.utils.hydroconfig import default_settings
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class TestAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

    # ====== Implementation of required methods ======

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
            tuned_parameters['max_depth'] = assignments['max_depth']
            tuned_parameters['learning_rate'] = assignments['learning_rate']
            tuned_parameters['min_split_loss'] = assignments['min_split_loss']
        else:
            tuned_parameters['max_depth'] = 11
            tuned_parameters['learning_rate'] = float(0.9294458527831317)
            tuned_parameters['min_split_loss'] = float(6.88375281543753)
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='test_sigopt.yaml'):
        config = {}
        config['project'] = 'e2eaiok'
        config['experiment'] = 'sklearn'
        parameters = [{
            'name': 'max_depth',
            'bounds': {
                'min': 3,
                'max': 12
            },
            'type': 'int'
        }, {
            'name': 'learning_rate',
            'bounds': {
                'min': 0.0,
                'max': 1.0
            },
            'type': 'double'
        }, {
            'name': 'min_split_loss',
            'bounds': {
                'min': 0.0,
                'max': 10
            },
            'type': 'double'
        }]
        user_defined_parameter = self.params['model_parameter'][
            'parameters'] if ('model_parameter' in self.params) and (
                'parameters' in self.params['model_parameter']) else None
        config['parameters'] = parameters
        if user_defined_parameter:
            self.logger.info(
                f"Update with user defined parameters {user_defined_parameter}"
            )
            update_list(config['parameters'], user_defined_parameter)
        config['metrics'] = [{
            'name': 'accuracy',
            'strategy': 'optimize',
            'objective': 'maximize'
        }, {
            'name': 'training_time',
            'objective': 'minimize'
        }]
        user_defined_metrics = self.params['model_parameter']['metrics'] if (
            'model_parameter' in self.params) and (
                'metrics' in self.params['model_parameter']) else None
        if user_defined_metrics:
            self.logger.info(
                f"Update with user defined parameters {user_defined_metrics}")
            update_list(config['metrics'], user_defined_metrics)
        config['observation_budget'] = self.params['observation_budget']

        # TODO: Add all parameter tuning here

        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def train_model(self, args):
        start_time = time.time()
        self.mean_accuracy, model_path = self.dist_launch(args)
        self.training_time = time.time() - start_time
        metrics = self.update_metrics()
        return self.training_time, model_path, metrics

    def dist_launch(self, args):
        # construct WnD launch command with mpi
        max_depth = args['model_parameter']["tuned_parameters"]['max_depth']
        learning_rate = args['model_parameter']["tuned_parameters"][
            'learning_rate']
        min_split_loss = args['model_parameter']["tuned_parameters"][
            'min_split_loss']
        model_saved_path = args['model_saved_path']
        cmd = []
        cmd.append(f"python")
        cmd.append(f"example/sklearn_train.py")
        cmd.append(f"--max_depth")
        cmd.append(f"{max_depth}")
        cmd.append(f"--learning_rate")
        cmd.append(f"{learning_rate}")
        cmd.append(f"--min_split_loss")
        cmd.append(f"{min_split_loss}")
        cmd.append(f"--saved_path")
        cmd.append(f"{model_saved_path}")
        self.logger.info(f'### Starting model training ###, launch cmd is: {" ".join(cmd)}')
        output = subprocess.check_output(cmd)
        return float(output), model_saved_path
