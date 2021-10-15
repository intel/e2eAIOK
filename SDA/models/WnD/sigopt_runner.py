from sigopt import Connection
import logging
import yaml
import time
import argparse
from multiprocessing import cpu_count
import subprocess
import os

def exclude_params(args, parameters):
    all_assignments = {}
    merged_parameters = []
    for param in parameters:
        all_assignments[param['name']] = param

    if len(args.deep_hidden_units) == 0:
        merged_parameters += [all_assignments['dnn_hidden_unit1'], all_assignments['dnn_hidden_unit2'], all_assignments['dnn_hidden_unit3']]
    if args.deep_learning_rate == -1:
        merged_parameters.append(all_assignments['deep_learning_rate'])
    if args.linear_learning_rate == -1:
        merged_parameters.append(all_assignments['linear_learning_rate'])
    if args.deep_warmup_epochs == -1:
        merged_parameters.append(all_assignments['deep_warmup_epochs'])
    if args.deep_dropout == -1:
        merged_parameters.append(all_assignments['deep_dropout'])
    return merged_parameters

def assign_params(args, assignments):
    if 'dnn_hidden_unit1' in assignments and 'dnn_hidden_unit2' in assignments and 'dnn_hidden_unit3' in assignments:
        args.deep_hidden_units = [assignments["dnn_hidden_unit1"], assignments["dnn_hidden_unit2"], assignments["dnn_hidden_unit3"]]
    if 'deep_learning_rate' in assignments:
        args.deep_learning_rate = assignments["deep_learning_rate"]
    if 'linear_learning_rate' in assignments:
        args.linear_learning_rate = assignments["linear_learning_rate"]
    if 'deep_warmup_epochs' in assignments:
        args.deep_warmup_epochs = assignments["deep_warmup_epochs"]
    if 'deep_dropout' in assignments:
        args.deep_dropout = assignments["deep_dropout"]

def create_experiment(args, data):
    conn = Connection()

    parameters = exclude_params(args, data["parameters"])
    metrics = data["metrics"]
    observation_budget = data["observation_budget"]
    experiment_name = data["experiment"]
    project = data["project"]

    experiment = conn.experiments().create(
        name=experiment_name,
        parameters=parameters,
        metrics=metrics,
        observation_budget=observation_budget,
        project=project,
    )
    return conn, experiment

def load_yaml(args, yaml_file="models/WnD/sigopt.yaml"):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    metrics = data["metrics"]
    args.optimize_training_time = False
    for metric in metrics:
        if (metric['name'] == 'training_time'):
            args.optimize_training_time = True
            break
    return data

def add_dataset_params(parser):
    group = parser.add_argument_group('datasets parameters')
    group.add_argument('--train_dataset_path', type=str, default='/outbrain/tfrecords/train/part*',
                       help='train dataset path')
    group.add_argument('--eval_dataset_path', type=str, default='/outbrain/tfrecords/eval/part*',
                       help='eval dataset path')
    group.add_argument('--dataset_meta_path', type=str, default='/outbrain/tfrecords/outbrain_meta.yaml',
                       help='Dataset metadata file')

def add_training_params(parser):
    group = parser.add_argument_group('training parameters')
    group.add_argument('--global_batch_size', type=int, default=131072,
                       help='Total size of training batch')
    group.add_argument('--eval_batch_size', type=int, default=131072,
                       help='Total size of evaluation batch')
    group.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    group.add_argument('--amp', default=False, action='store_true',
                       help='Enable automatic mixed precision conversion')
    group.add_argument('--xla', default=False, action='store_true',
                       help='Enable XLA conversion')
    group.add_argument('--linear_learning_rate', type=float, default=-1,
                       help='Learning rate for linear model')
    group.add_argument('--deep_learning_rate', type=float, default=-1,
                       help='Learning rate for deep model')
    group.add_argument('--deep_warmup_epochs', type=float, default=-1,
                       help='Number of learning rate warmup epochs for deep model')
    group.add_argument('--model_dir', type=str, default='/outbrain/checkpoints',
                       help='Destination where model checkpoint will be saved')
    group.add_argument('--metric', type=str, default='AUC',
                       help='Evaluation metric')
    group.add_argument('--metric_threshold', type=float, default=0,
                       help='Metric threshold used for training early stop')

def add_model_params(parser):
    group = parser.add_argument_group('model construction parameters')
    group.add_argument('--deep_hidden_units', type=int, default=[], nargs="+",
                       help='Hidden units per layer for deep model, separated by spaces')
    group.add_argument('--deep_dropout', type=float, default=-1,
                       help='Dropout regularization for deep model')

def add_distributed_training_params(parser):
    group = parser.add_argument_group('distributed training parameters')
    group.add_argument('--hosts', type=str, default=['localhost'], nargs='+',
                       help='Hosts to launch training, separated by spaces')
    group.add_argument('--ppn', type=int, default=1,
                       help='Define worker number per node for distributed training')
    group.add_argument('--cores', type=int, default=cpu_count(),
                       help='Define worker number per node for distributed training')
    group.add_argument('--ccl_worker_num', type=int, default=1,
                       help='Define worker number per node for distributed training')

def parse_args():
    parser = argparse.ArgumentParser(
        description='WideAndDeep Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    add_dataset_params(parser)
    add_training_params(parser)
    add_model_params(parser)
    add_distributed_training_params(parser)

    parser.add_argument('--program', type=str, required=True,
                        help='The full path to the proram/script to be launched')
    parser.add_argument('--python_executable', type=str, default='python', 
                        help='Python interpreter')
    args, _ = parser.parse_known_args()
    return args

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
    cmd = f"time mpirun -genv OMP_NUM_THREADS={omp_threads} -map-by socket -n {ranks} -ppn {ppn} -hosts {','.join(hosts)} -print-rank-map "
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

def train_model(args):
    start_time = time.time()
    if args.ppn == 1 and len(args.hosts) == 1:
        launch(args)
    else:
        dist_launch(args)
    training_time = time.time() - start_time
    return training_time

def main():
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('sigopt')
    args = parse_args()

    experiment_data = load_yaml(args)
    conn, experiment = create_experiment(args, experiment_data)
    budget = experiment.observation_budget
    logger.info(f'Experiment budget: {budget}')

    for _ in range(budget):
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments = suggestion.assignments

        logger.info(f'Assignment: {assignments}')
        assign_params(args, assignments)
        logger.info(f'The training parameters from sigopt: dnn_hidden_units: {args.deep_hidden_units}, linear_learning_rate: {args.linear_learning_rate}, \
            deep_learning_rate: {args.deep_learning_rate}, deep_warmup_epochs: {args.deep_warmup_epochs}, deep_dropout: {args.deep_dropout}')

        training_time = train_model(args)
        values = []
        metric_file = os.path.join(os.path.dirname(os.path.abspath(args.program)), 'metric.txt')
        with open(metric_file) as f:
            lines = f.readlines()
        values.append({'name': args.metric, 'value': float(lines[-1])})
        if args.optimize_training_time:
            values.append({'name': 'training_time', 'value': training_time})
        logger.info(f'Sigopt observation values: {values}')

        for i in range(5):
            try:
                conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion.id,
                    values=values,
                )
                experiment = conn.experiments(experiment.id).fetch()
                break
            except Exception as e:
                logger.info(f'Met exception when creating observation, retried {i} times. The exception is: {e}')
                time.sleep(5)

    all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
    if len(all_best_assignments.data) == 0:
        logger.info("No assignments for satisfied model, you may increase observation budget or modify metric value")
    else:
        logger.info(f"Best Assignments: {all_best_assignments.data}")
    logger.info(f"Please go to https://app.sigopt.com/experiment/{experiment.id} for optimization history and analysis")


def train_wo_sigopt():
    args = parse_args()
    metrics = train_model(args)

if __name__ == '__main__':
    main()
