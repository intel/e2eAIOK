from trainer.model.widedeep import wide_deep_model
from trainer.run import train
from trainer.utils.arguments import parse_args
from trainer.utils.setup import create_config
import horovod.tensorflow as hvd

from sigopt import Connection
import logging
import yaml
import time

def create_experiment(yaml_file="sigopt.yaml"):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    conn = Connection()
    parameters = data["sigopt"]["parameters"]
    metrics = data["sigopt"]["metrics"]
    observation_budget = data["sigopt"]["observation_budget"]
    experiment_name = data["sigopt"]["experiment"]
    project = data["sigopt"]["project"]
    parallel_bandwidth = data["sigopt"]["parallel_bandwidth"]
    experiment = conn.experiments().create(
        name=experiment_name,
        parameters=parameters,
        metrics=metrics,
        parallel_bandwidth=parallel_bandwidth,
        observation_budget=observation_budget,
        project=project,
    )
    return conn, experiment


def main():
    args = parse_args()
    config = create_config(args)

    logger = logging.getLogger('tensorflow')

    assignments = {}
    experiment_id = 0
    suggestion_id = 0
    if hvd.rank() == 0:
        conn, experiment = create_experiment()
        experiment_id = experiment.id
        experiment_id = hvd.broadcast_object(experiment_id, root_rank=0)
        logger.info(f'experiment.observation_budget: {experiment.observation_budget}')
    else:
        experiment_id = hvd.broadcast_object(experiment_id, root_rank=0)

    conn = Connection()
    experiment = conn.experiments(experiment_id).fetch()
    for _ in range(experiment.observation_budget):

        if hvd.rank() == 0:
            suggestion = conn.experiments(experiment.id).suggestions().create()
            assignments = suggestion.assignments
            assignments = hvd.broadcast_object(assignments, root_rank=0)
        else:
            assignments = hvd.broadcast_object(assignments, root_rank=0)
        logger.info(f'{hvd.rank()}:assignment: {assignments}')
        args.deep_hidden_units = [assignments["dnn_hidden_unit1"], assignments["dnn_hidden_unit2"], assignments["dnn_hidden_unit3"]]
        args.deep_learning_rate = assignments["deep_learning_rate"]
        args.linear_learning_rate = assignments["linear_learning_rate"]
        args.deep_warmup_epochs = assignments["deep_warmup_epochs"]
        args.deep_dropout = assignments["deep_dropout"]

        model = wide_deep_model(args, config['features'])
        map = train(args, model, config)
        if hvd.rank() == 0:
            for i in range(5):
                try:
                    conn.experiments(experiment.id).observations().create(
                        suggestion=suggestion.id,
                        value=map,
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

if __name__ == '__main__':
    main()
