import subprocess #nosec
import yaml
import logging
import time

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class UPMAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

        # set default required arguments
        self.params[
            'ppn'] = 1 if 'ppn' not in self.params else self.params['ppn']
        self.params[
            'num_instances'] = 1 if 'num_instances' not in self.params else self.params[
                'num_instances']
        self.params[
            'num_cores'] = 1 if 'num_cores' not in self.params else self.params[
                'num_cores']
        self.params['metric'] = {
            'name': 'AUC',
            'threshold': 0.6
        } if 'metric' not in self.params else self.params['metric']

        # check distributed configuration
        missing_params = []
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
        self.saved_path = self.params['save_path']
        if self.params['platform'] == 'tensorflow':
            self.train_python = "/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python"
        elif self.params['platform'] == 'pytorch':
            self.train_python = "/opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/bin/python"
        self.train_script = "/home/vmagent/app/e2eaiok/modelzoo/upm/main.py"

    def update_metrics(self):
        result_metrics_file = os.path.join(self.saved_path, "metric.txt")
        if not os.path.exists(result_metrics_file):
            raise FileNotFoundError(
                f"{self.train_script} completed, while we can't find \
                    result {result_metrics_file} file.")
        with open(result_metrics_file) as f:
            metric = f.readlines()
        metrics = []
        metrics.append({
            'name': self.params['metric']['name'],
            'value': float(metric[-1])
        })
        metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments=None):
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters['learning_rate'] = assignments[
                'learning_rate']
        else:
            tuned_parameters['learning_rate'] = float(0.01)
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='test_sigopt.yaml'):
        config = {}
        config['project'] = self.params['model_parameter']['project']
        config['experiment'] = self.params['model_parameter']['experiment']
        parameters = [{
            'name': 'learning_rate',
            'bounds': {
                'min': 1.0e-4,
                'max': 1.0e-1
            },
            'type': 'double',
            'transformation': 'log'
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
            'objective': 'maximize',
            'threshold': 0.6
        }, {
            'name': 'training_time',
            'objective': 'minimize',
            'threshold': 1800
        }]
        user_defined_metrics = self.params['model_parameter']['metrics'] if (
            'model_parameter' in self.params) and (
                'metrics' in self.params['model_parameter']) else None
        if user_defined_metrics:
            self.logger.info(
                f"Update with user defined parameters {user_defined_metrics}")
            update_list(config['metrics'], user_defined_metrics)
        config['observation_budget'] = self.params['observation_budget']

        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def train_model(self, args):
        start_time = time.time()
        if args['ppn'] > 1:
            self.dist_launch(args)
        else:
            self.launch(args)
        self.training_time = time.time() - start_time
        metrics = self.update_metrics()
        model_path = args['model_saved_path']
        return self.training_time, model_path, metrics

    def dist_launch(self, args):
        cores = args['cores']
        ppn = args['ppn']
        ccl_worker_num = args['ccl_worker_num']
        hosts = args['hosts']
        omp_threads = cores // 2 // ppn - ccl_worker_num
        ranks = len(hosts) * ppn

        # construct WnD launch command with mpi
        if self.params['platform'] == 'tensorflow':
            os.environ['CCL_WORKER_COUNT'] = str(ccl_worker_num)
            cmd = f"mpirun -genv OMP_NUM_THREADS={omp_threads} -map-by socket -n {ranks} -ppn {ppn} -hosts {','.join(hosts)} -print-rank-map "
            cmd += f"-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 "
            cmd += f"{self.train_python} -u "
        elif self.params['platform'] == 'pytorch':
            cmd = f"{self.train_python} -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node={ppn} --nnodes={len(hosts)} --hostfile {','.join(hosts)} "

        cmd += f"{self.train_script} "
        cmd += f"--results_dir {self.saved_path} "
        cmd += f"--train_data_pattern '{self.train_path}' "
        cmd += f"--eval_data_pattern '{self.test_path}' "
        cmd += f"--dataset_meta_file {self.dataset_meta_path} "
        cmd += f"--global_batch_size {args['global_batch_size']} "
        cmd += f"--eval_batch_size {args['global_batch_size']} "
        cmd += f"--num_epochs {args['num_epochs']} "
        cmd += f"--metric {args['metric']['name']} "
        cmd += f"--metric_threshold {args['metric']['threshold']} "
        cmd += f"--platform {self.params['platform']} "
        cmd += f"--model_dir {args['model_dir']} "
        cmd += f"--learning_rate {args['model_parameter']['tuned_parameters']['learning_rate']} "
        self.logger.info(f'training launch command: {cmd}')

        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def launch(self, args):
        # construct WnD launch command
        if self.params['platform'] == 'tensorflow':
            cmd = f"{self.train_python} -u "
        elif self.params['platform'] == 'pytorch':
            cmd = f"{self.train_python} -m intel_extension_for_pytorch.cpu.launch --throughput_mode "
        cmd += f"{self.train_script} "
        cmd += f"--results_dir {self.saved_path} "
        cmd += f"--train_data_pattern '{self.train_path}' "
        cmd += f"--eval_data_pattern '{self.test_path}' "
        cmd += f"--dataset_meta_file {self.dataset_meta_path} "
        cmd += f"--global_batch_size {args['global_batch_size']} "
        cmd += f"--eval_batch_size {args['global_batch_size']} "
        cmd += f"--num_epochs {args['num_epochs']} "
        cmd += f"--metric {args['metric']['name']} "
        cmd += f"--metric_threshold {args['metric']['threshold']} "
        cmd += f"--platform {self.params['platform']} "
        cmd += f"--model_dir {args['model_dir']} "
        cmd += f"--learning_rate {args['model_parameter']['tuned_parameters']['learning_rate']} "
        self.logger.info(f'training launch command: {cmd}')

        process = subprocess.Popen(cmd, shell=True)
        process.wait()
