import subprocess
import yaml
import logging
import time
import os

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class MiniGoAdvisor(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')
        self.train_path = train_path
        self.test_path = eval_path
        self.dataset_meta_path = dataset_meta_path
        self.data_path = settings['data_path']
        self.checkpoint_dir = f"{settings['data_path']}/checkpoints/mlperf07"
        self.target=f"{settings['data_path']}/target/target.minigo"

    def update_metrics(self):
        with open("result/metric.txt", "r") as f:
            self.winrate = float(f.read())
        metrics = []
        metrics.append({'name': 'winrate', 'value': self.winrate})
        metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments=None):
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters['train_batch_size'] = assignments[
                'train_batch_size']
        else:
            tuned_parameters['train_batch_size'] = 4096
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='minigo_sigopt.yaml'):
        config = {}
        config['project'] = self.params['model_parameter']['project']
        config['experiment'] = self.params['model_parameter']['experiment']
        parameters = [
        {'name':'train_batch_size','grid':[512,1024,2048,4096,8192],'type':'int'}
        ]
        config['parameters'] = parameters
        config['metrics'] = [{
            'name': 'winrate',
            'objective': 'maximize',
        }, {
            'name': 'training_time',
            'strategy': 'optimize',
            'objective': 'minimize',
        }]
        config['observation_budget'] = self.params['observation_budget']

        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def train_model(self, args):
        start_time = time.time()
        self.dist_launch(args)
        self.training_time = time.time() - start_time
        metrics = self.update_metrics()
        model_path = args['model_saved_path']
        return self.training_time, model_path, metrics

    def dist_launch(self, args):
        hosts = args['hosts']
        rootnode = args['rootnode']
        os.chdir('modelzoo/minigo')
        pre_cmd=f"sed -i '/--train_batch_size=/ s/=.*/={args['model_parameter']['tuned_parameters']['train_batch_size']}/' ml_perf/flags/19/train.flags"
        process = subprocess.Popen(pre_cmd, shell=True)
        process.wait()
        cmd = f"HOSTLIST={','.join(hosts)} ROOTNODE={rootnode} checkpoint_dir={self.checkpoint_dir} target={self.target} ml_perf/scripts/run_minigo.sh 19"
        self.logger.info(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        clean_cmd = f"pkill -f ml_perf"
        process = subprocess.Popen(clean_cmd, shell=True)
        process.wait()
        os.chdir('../../')
