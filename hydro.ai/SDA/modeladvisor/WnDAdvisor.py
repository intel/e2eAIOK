import subprocess
import yaml
import logging

from modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class WnDAdvisor(BaseModelAdvisor):
    '''
        Wide and Deep sigopt model optimization launcher
    '''
    def __init__(self, dataset_meta_path, train_path, eval_path, args):
        super().__init__(dataset_meta_path, train_path, eval_path, args)
        params = self.init_WnDAdvisor_params()
        params.update(self.params)
        self.params = params

    def init_WnDAdvisor_params(self):
        '''
            Add model specific parameters
            For sigopt parameters, set parameter explicitly and the parameter will not be optimized by sigopt
        '''
        params = {}
        # model params
        params['deep_hidden_units'] = 4096
        params['deep_hidden_units'] = []
        params['deep_dropout'] = float(1)
        # train params
        params['linear_learning_rate'] = float(-1)
        params['deep_learning_rate'] = float(-1)
        params['deep_warmup_epochs'] = float(-1)
        return params
    
    ###### Implementation of required methods ######

    def generate_sigopt_yaml(self, file='wnd_sigopt.yaml'):
        config = {}
        config['project'] = 'HYDROAI'
        config['experiment'] = 'WnD'
        parameters = [{'name': 'dnn_hidden_unit1', 'grid': [64, 128, 256, 512, 1024, 2048], 'type': 'int'}, 
                      {'name': 'dnn_hidden_unit2', 'grid': [64, 128, 256, 512, 1024, 2048], 'type': 'int'}, 
                      {'name': 'dnn_hidden_unit3', 'grid': [64, 128, 256, 512, 1024, 2048], 'type': 'int'}, 
                      {'name': 'deep_learning_rate', 'bounds': {'min': 1.0e-4, 'max': 1.0e-1}, 'type': 'double', 'transformation': 'log'},
                      {'name': 'linear_learning_rate', 'bounds': {'min': 1.0e-2, 'max': 1.0}, 'type': 'double', 'transformation': 'log'}, 
                      {'name': 'deep_warmup_epochs', 'bounds': {'min': 1, 'max': 8}, 'type': 'int'},
                      {'name': 'deep_dropout', 'bounds': {'min': 0, 'max': 0.5}, 'type': 'double'}]
        config['parameters'] = parameters
        config['metrics'] = self.params['metrics']
        config['observation_budget'] = self.params['observation_budget']

        # TODO: Add all parameter tuning here

        with open(file, 'w') as f:
            yaml.dump(config, f)
        return config

    def update_metrics(self):
        # we should update through trained result with default
        return self.params['metrics']
  
    def train_model(self, args):
        start_time = time.time()
        if args.ppn == 1 and len(args.hosts) == 1:
            launch(args)
        else:
            dist_launch(args)
        training_time = time.time() - start_time
        return training_time

    def launch(args):
        # construct WnD launch command
        cmd = f"{args.python_executable} -u {args.program} "
        cmd += f"--train_data_pattern '{args.train_dataset_path}' --eval_data_pattern '{args.eval_dataset_path}' --dataset_meta_file {args.dataset_meta_path} " \
            + f"--global_batch_size {args.global_batch_size} --eval_batch_size {args.eval_batch_size} --num_epochs {args.num_epochs} " \
            + f"--metric {args.metric} --metric_threshold {args.metric_threshold} "
        if args.linear_learning_rate != -1:
            cmd += f"--linear_learning_rate {args.linear_learning_rate} "
        if args.deep_learning_rate != -1:
            cmd += f"--deep_learning_rate {args.deep_learning_rate} "
        if args.deep_warmup_epochs != -1:
            cmd += f"--deep_warmup_epochs {args.deep_warmup_epochs} "
        if len(args.deep_hidden_units) != 0:
            cmd += f"--deep_hidden_units {' '.join([str(item) for item in args.deep_hidden_units])} "
        if args.deep_dropout != -1:
            cmd += f"--deep_dropout {args.deep_dropout} "
        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
    
    def dist_launch(args):
        cores = args.cores
        ppn = args.ppn
        ccl_worker_num = args.ccl_worker_num
        hosts = args.hosts
        omp_threads = cores // 2 // ppn - ccl_worker_num
        ranks = len(hosts) * ppn
    
        # construct WnD launch command with mpi
        cmd = f"mpirun -genv OMP_NUM_THREADS={omp_threads} -map-by socket -n {ranks} -ppn {ppn} -hosts {','.join(hosts)} -print-rank-map "
        cmd += f"-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 "
        cmd += f"{args.python_executable} -u {args.program} "
        cmd += f"--train_data_pattern '{args.train_dataset_path}' --eval_data_pattern '{args.eval_dataset_path}' --dataset_meta_file {args.dataset_meta_path} " \
            + f"--global_batch_size {args.global_batch_size} --eval_batch_size {args.eval_batch_size} --num_epochs {args.num_epochs} " \
            + f"--metric {args.metric} --metric_threshold {args.metric_threshold} "
        if args.linear_learning_rate != -1:
            cmd += f"--linear_learning_rate {args.linear_learning_rate} "
        if args.deep_learning_rate != -1:
            cmd += f"--deep_learning_rate {args.deep_learning_rate} "
        if args.deep_warmup_epochs != -1:
            cmd += f"--deep_warmup_epochs {args.deep_warmup_epochs} "
        if len(args.deep_hidden_units) != 0:
            cmd += f"--deep_hidden_units {' '.join([str(item) for item in args.deep_hidden_units])} "
        if args.deep_dropout != -1:
            cmd += f"--deep_dropout {args.deep_dropout} "
        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()       
