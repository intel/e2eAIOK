import logging
import subprocess
import yaml

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class DIENAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')
        self.total_cores, self.physical_cores = self.get_cpu_info()

        # set default required arguments
        if 'ppn' not in self.params:
            self.params['ppn'] = 1
        self.params['num_instances'] = self.physical_cores
        self.params['num_cores'] = int(self.total_cores / self.physical_cores)

        # check distributed configuration
        missing_params = []
        # mpirun -n 1 -hosts 172.16.8.30 -ppn 1 -iface ens21f1
        if self.params["ppn"] > 1:
            missing_params = missing_params + \
                ['hosts'] if 'hosts' not in self.params else missing_params
            missing_params = missing_params + \
                ['iface'] if 'iface' not in self.params else missing_params
        if len(missing_params) > 0:
            raise ValueError(
                f"[CONFIG ERROR] Missing parameters {missing_params} in \
                e2eaiok_defaults.conf when ppn is set above 1.")

        self.train_path = train_path
        self.test_path = eval_path
        self.dataset_meta_path = dataset_meta_path
        # self.saved_path = settings['model_saved_path']
        self.python_path = "/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/"
        self.train_python = f"{self.python_path}/python"
        self.horovodrun_path = f"{self.python_path}/horovodrun"
        self.train_script = "/home/vmagent/app/e2eaiok/modelzoo/dien/train/ai-matrix/script/train.py"

    def get_cpu_info(self):
        # get cpu physical cores and virtual cores per core as return
        num_total_cores = 128
        num_physical_cores = 32
        return num_total_cores, num_physical_cores

    # ====== Implementation of required methods ======

    def update_metrics(self):
        result_metrics_path = os.path.join(self.params['model_saved_path'],
                                           "result.yaml")
        if not os.path.exists(result_metrics_path):
            raise FileNotFoundError(
                f"{self.train_script} completed, while we can't find \
                    result {result_metrics_path} file.")
        with open(result_metrics_path) as f:
            results = yaml.load(f, Loader=yaml.FullLoader)
        self.training_time = results['training_time']
        self.best_trained_model_path = results['best_trained_model']
        metrics = []
        metrics.append({'name': 'AUC', 'value': results['AUC']})
        metrics.append({
            'name': 'training_time',
            'value': results['training_time']
        })
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments=None):
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters['batch_size'] = assignments['batch_size']
        else:
            tuned_parameters['batch_size'] = 256
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='test_sigopt.yaml'):
        config = {}
        config['project'] = 'e2eaiok'
        config['experiment'] = 'dien'
        parameters = [{
            'name': 'batch_size',
            'categorical_values': ["256", "512", "1024"],
            'type': 'categorical'
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
            'name': 'AUC',
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
        if args['ppn'] > 1:
            self.dist_launch(args)
        else:
            self.launch(args)
        metrics = self.update_metrics()
        return self.training_time, self.best_trained_model_path, metrics

    def dist_launch(self, args):
        cmd = []
        hosts = [f"{h}:1" for h in args['hosts']]
        cmd.extend([self.horovodrun_path, "-np", f"{args['ppn']}", "-H",  f"{','.join(hosts)}", "--network-interface", f"{args['iface']}"])
        cmd.extend(["--verbose"])
        cmd.extend(self.prepare_cmd(args))

        self.logger.info(f'training launch command: {" ".join(cmd)}')
        process = subprocess.Popen(cmd)
        process.wait()

    def launch(self, args):
        cmd = []
        cmd.extend(self.prepare_cmd(args))
        self.logger.info(f'training launch command: {" ".join(cmd)}')
        process = subprocess.Popen(cmd)
        process.wait()

    def prepare_cmd(self, args):
        cmd = []
        cmd.extend([
            f"{self.train_python}", "-u", f"{self.train_script}", "--train_path",
            f"{self.train_path}", "--test_path", f"{self.test_path}",
            "--meta_path", f"{self.dataset_meta_path}"
        ])
        cmd.extend(["--saved_path", f"{args['model_saved_path']}"])
        cmd.extend(["--num-intra-threads", f"{args['num_instances']}"])
        cmd.extend(["--num-inter-threads", f"{args['num_cores']}"])

        # fixed parameters
        cmd.extend([
            "--mode", "train", "--embedding_device", "cpu", "--model", "DIEN"
        ])
        cmd.extend(["--slice_id", "0", "--advanced", "true", "--seed", "3"])
        cmd.extend(["--data_type", "FP32"])

        # tunnable parameters
        tuned_parameters = args['model_parameter']['tuned_parameters']
        cmd.extend(["--batch_size", f"{tuned_parameters['batch_size']}"])

        # fake result
        # result_metrics_path = os.path.join(self.params['model_saved_path'], "result.yaml")
        # result = {"AUC": 0.823, "training_time": 425, "best_trained_model": f"{self.params['model_saved_path']}/best_trained_model"}
        # with open(result_metrics_path, "w") as f:
        #    results = yaml.dump(result, f)
        return cmd
