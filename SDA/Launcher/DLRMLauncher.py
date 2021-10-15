import argparse
import subprocess
import yaml
from Launcher.BaseModelLauncher import BaseModelLauncher

class DLRMLauncher(BaseModelLauncher):
    def __init__(self, dataset_meta_path, train_path, eval_path, args):
        super().__init__(dataset_meta_path, train_path, eval_path, args)
        args = self.parse_args(args)

        self.params['test_mini_batch_size'] = args.test_mini_batch_size
        self.params['print_freq'] = args.print_freq
        self.params['test_freq'] = args.test_freq
        self.params['day_feature_count'] = args.day_feature_count
        
        self.generate_sigopt_yaml(file = 'models/DLRM/sigopt.yaml')

    def parse_args(self, args):

        self.parser.add_argument("--test-mini-batch-size", type=int, default=32768)
        self.parser.add_argument("--print-freq", type=int, default=16)
        self.parser.add_argument("--test-freq", type=int, default=800)
        self.parser.add_argument("--day-feature-count", type=str, default="./data/day_fea_count.npz")
        return self.parser.parse_args(args)
    
    def generate_sigopt_yaml(self, file):
        config = {}
        config['project'] = 'dlrm'
        config['experiment'] = 'DLRM'
        parameters = [{'name':'learning_rate','bounds':{'min':5,'max':50},'type':'int'},
                      {'name':'lamb_lr','bounds':{'min':5,'max':50},'type':'int'},
                      {'name':'warmup_steps','bounds':{'min':2000,'max':4500},'type':'int'},
                      {'name':'decay_start_steps','bounds':{'min':4501,'max':9000},'type':'int'},
                      {'name':'num_decay_steps','bounds':{'min':5000,'max':15000},'type':'int'},
                      {'name':'sparse_feature_size','grid': [128,64,16],'type':'int'},
                      {'name':'mlp_top_size','bounds':{'min':0,'max':7},'type':'int'},
                      {'name':'mlp_bot_size','bounds':{'min':0,'max':3},'type':'int'},
                      {'name':'bf16','bounds':{'min':0,'max':1},'type':'int'}]
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
        python_executable = self.params['python_executable']
        ppn = self.params['ppn'] if 'ppn' in self.params else 1
        # hosts_array = np.fromstring(self.params['hosts'], dtype=int, sep=",")
        hosts = self.params['hosts']
        num_nodes = len(hosts)

        cmd = f"{python_executable} models/DLRM/sigopt_runner.py "
        cmd += f"--distributed --nproc_per_node={ppn} --nnodes={num_nodes} --hostfile {','.join(hosts)} " \
            + f"../examples/dlrm/dlrm/dlrm_s_pytorch.py --mini-batch-size={self.params['global_batch_size']} --print-freq={self.params['print_freq']} " \
            + f"--test-mini-batch-size={self.params['test_mini_batch_size']} --test-freq={self.params['test_freq']} " \
            + f"--train-data-path={self.params['train_dataset_path']} --eval-data-path={self.params['eval_dataset_path']} " \
            + f"--nepochs={self.params['num_epochs']} --day-feature-count={self.params['day_feature_count']} " \
            + f"--loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex " \
            + f" --dist-backend=ccl --print-time --data-generation=dataset --optimizer=1 --bf16 --data-set=terabyte " \
            + f"--sparse-dense-boundary=403346  --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 " \
            + f"--mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=12345"
        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

#--master_addr={hosts[0]} 