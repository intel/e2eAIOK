from trainer.model.widedeep import wide_deep_model
from trainer.run import train
from trainer.utils.arguments import parse_args
from trainer.utils.setup import create_config
import horovod.tensorflow as hvd

from sigopt import Connection
import logging
import yaml
import time

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

def main():
    args = parse_args()
    config = create_config(args)
    logger = logging.getLogger('tensorflow')

    experiment_data = load_yaml(args)
    assignments = {}
    budget = 0
    if hvd.rank() == 0:
        conn, experiment = create_experiment(args, experiment_data)
        budget = experiment.observation_budget
        budget = hvd.broadcast_object(budget, root_rank=0)
    else:
        budget = hvd.broadcast_object(budget, root_rank=0)
    logger.info(f'Experiment budget: {budget}')

    for _ in range(budget):

        if hvd.rank() == 0:
            suggestion = conn.experiments(experiment.id).suggestions().create()
            assignments = suggestion.assignments
            assignments = hvd.broadcast_object(assignments, root_rank=0)
        else:
            assignments = hvd.broadcast_object(assignments, root_rank=0)
        logger.info(f'{hvd.rank()}:assignment: {assignments}')
        assign_params(args, assignments)
        logger.info(f'The training parameters from sigopt: dnn_hidden_units: {args.deep_hidden_units}, linear_learning_rate: {args.linear_learning_rate}, \
            deep_learning_rate: {args.deep_learning_rate}, deep_warmup_epochs: {args.deep_warmup_epochs}, deep_dropout: {args.deep_dropout}')

        model = wide_deep_model(args)
        metrics = train(args, model, config)
        values = []
        values.append({'name': args.metric, 'value': metrics['metric']})
        if args.optimize_training_time:
            values.append({'name': 'training_time', 'value': metrics['training_time']})
        logger.info(f'Sigopt observation values: {values}')
        if hvd.rank() == 0:
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

    if hvd.rank() == 0:
        all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
        best_assignments = all_best_assignments.data[0].assignments
        logger.info("Best Assignments: " + str(best_assignments))
        logger.info(f"Please go to https://app.sigopt.com/experiment/{experiment.id} for optimization history and analysis")


def train_wo_sigopt():
    args = parse_args()
    config = create_config(args)

    logger = logging.getLogger('tensorflow')

    model = wide_deep_model(args)
    metrics = train(args, model, config)

if __name__ == '__main__':
    main()
