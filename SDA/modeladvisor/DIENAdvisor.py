import subprocess
import yaml
import logging
import time
import sys
from common.utils import *

from SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class DIENAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

        # set default required arguments
        self.params['ppn'] = 1 if 'ppn' not in self.params else self.params['ppn']
        self.params['num_instances'] = 24 if 'num_instances' not in self.params else self.params['num_instances']
        self.params['num_cores'] = 4 if 'num_cores' not in self.params else self.params['num_cores']

        # check distributed configuration
        missing_params = []
        # mpirun -n 1 -hosts 172.16.8.30 -ppn 1 -iface ens21f1
        if self.params["ppn"] > 1:
            missing_params = missing_params + ['hosts'] if 'hosts' not in self.params else missing_params
            missing_params = missing_params + ['iface'] if 'iface' not in self.params else missing_params
        if len(missing_params) > 0:
            raise ValueError(f"[CONFIG ERROR] Missing parameters {missing_params} in hydroai_defaults.conf when ppn is set above 1.")

        self.train_path = train_path
        self.test_path = eval_path
        self.saved_path = args['model_saved_path']
        self.train_python = "/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python"
        self.train_script = "/home/vmagent/app/hydro.ai/in-stock-models/dien/train/ai-matrix/horovod/script/train.py"
    
    ###### Implementation of required methods ######

    def update_metrics(self):
        result_metrics_path = os.path.join(self.saved_path, "result.yaml")
        if not os.path.exists(result_metrics_path):
            raise FileNotFoundError(f"{self.train_script} completed, while we can't find result {result_metrics_path} file.")
        with open(self.saved_path) as f:
            results = yaml.load(f, Loader=yaml.FullLoader)
        metrics = []
        metrics.append({'name': 'AUC', 'value': results['AUC']})
        metrics.append({'name': 'training_time', 'value': results['training_time']})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments = None):
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
        self.params['model_saved_path'] = os.path.join(self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config
    
    def generate_sigopt_yaml(self, file='test_sigopt.yaml'):
        config = {}
        config['project'] = 'hydro.ai'
        config['experiment'] = 'sklearn'
        parameters = [{'name': 'max_depth', 'bounds': {'min': 3, 'max': 12}, 'type': 'int'}, 
                      {'name': 'learning_rate', 'bounds': {'min': 0.0, 'max': 1.0}, 'type': 'double'},
                      {'name': 'min_split_loss', 'bounds': {'min': 0.0, 'max': 10}, 'type': 'double'}]
        user_defined_parameter = self.params['model_parameter']['parameters'] if ('model_parameter' in self.params) and ('parameters' in self.params['model_parameter']) else None
        config['parameters'] = parameters
        if user_defined_parameter:
            self.logger.info(f"Update with user defined parameters {user_defined_parameter}")
            update_list(config['parameters'], user_defined_parameter)
        config['metrics'] = [
            {'name': 'accuracy', 'strategy': 'optimize', 'objective': 'maximize'},
            {'name': 'training_time', 'objective': 'minimize'}
        ]
        user_defined_metrics = self.params['model_parameter']['metrics'] if ('model_parameter' in self.params) and ('metrics' in self.params['model_parameter']) else None
        if user_defined_metrics:
            self.logger.info(f"Update with user defined parameters {user_defined_metrics}")
            update_list(config['metrics'], user_defined_metrics)
        config['observation_budget'] = self.params['observation_budget']

        # TODO: Add all parameter tuning here

        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config
  
    def train_model(self, args):
        if args['ppn'] > 1:
            self.dist_launch(args)
        else:
            self.launch(args)
        metrics = self.update_metrics()
        return self.training_time, model_path, metrics
    
    def dist_launch(self, args):
        cmd = []
        # mpirun -n 1 -hosts 172.16.8.30 -ppn 1 -iface ens21f1 -print-rank-map -prepend-rank -verbose 
        cmd.extend(["mpirun", "-n", f"{args['ppn']}", "-hosts", f"{args['hosts']}", "-iface", f"{args['iface']}"])
        cmd.extend(["-print-rank-map", "-prepend-rank", "-verbose"])
        cmd.extend(self.prepare_cmd(args))

        self.logger.info(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd)
        process.wait()

    def prepare_cmd(self, args):
        cmd = []
        cmd.extend([f"{self.train_python}", f"{self.train_script}", "--train_path", f"{self.train_path}", "--test_path", f"{self.test_path}"])
        cmd.append()
        cmd.append()
        cmd.append()
        cmd.append(f)
        cmd.append()
        cmd.append(f"--saved_path")
        cmd.append(f"{model_saved_path}")
        cmd.append(f"--mode")
        cmd.append(f"train")
        cmd.append(f"--batch_size")
        cmd.append(f"1024")
        cmd.append(f"--num-intra-threads")
        cmd.append(f"{args['num_instances']}")
        cmd.append(f"--num-inter-threads")
        cmd.append(f"{args['num_cores']}")
        return cmd
        

    def launch(self, args):
        cmd = []
        cmd.append(f"/opt/intel/oneapi/intelpython/latest/bin/python")
        cmd.append(f"/home/vmagent/app/hydro.ai/example/sklearn_train.py")
        cmd.append(f"--train_path")
        cmd.append(f"{self.train_path}")
        cmd.append(f"--test_pR")
        cmd.append(f"{learning_rate}")
        cmd.append(f"--min_split_loss")
        cmd.append(f"{min_split_loss}")
        cmd.append(f"--saved_path")
        cmd.append(f"{model_saved_path}")
        self.logger.info(f'training launch command: {cmd}')
        output = subprocess.check_output(cmd)
        return float(output), model_saved_path
