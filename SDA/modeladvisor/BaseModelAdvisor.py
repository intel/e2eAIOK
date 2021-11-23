import argparse
import logging
from sigopt import Connection
import time
from hydroai.hydroconfig import *
from common.utils import *
import yaml
import sys
import json
from collections import OrderedDict

class BaseModelAdvisor:
    """Model Advisor Base, Model Advisor is used to create w/wo sigopt parameter advise based on model type

    ...

    Attributes
    ----------
    conn
        sigopt connection
    experiment
        current experiment which is either created by this run or
        fetched by history experiment id
    params : dict
        params include dataset_path, save_path, global_configs,
        model_parameters, passed by arguments or hydroai-defaults.conf
    assignment_model_tracker : dict
        a tracker map of assigned_parameters and its corresponding
        model path

    Methods
    -------
    generate_sigopt_yaml()
        generate parameter range based on model type and passed-in
        params for sigopt 
    initialize_model_parameter(assignments = None)
        generate parameters based on model type for wo sigopt case
    train_model(args)
        launch train with passed-in model parameters
    update_metrics()
        update return metrics based on training result, normally
        training script should have a stdout output or a result.yaml
        file, and this method will pass this file and return metrics
    """
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

        self.conn = None
        self.experiment = None
        self.params = init_advisor_params()
        self.params['dataset_meta_path'] = dataset_meta_path
        self.params['train_dataset_path'] = train_path
        self.params['eval_dataset_path'] = eval_path
        dest_saved_path = os.path.join(self.params['save_path'], settings['model_name'])
        self.params['save_path'] = mkdir(dest_saved_path)
        self.assignment_model_tracker = {}
        self.params.update(settings)

    def initialize_sigopt(self, experiment_id = None):
        # 1. create sigopt yaml
        sigopt_config = self.generate_sigopt_yaml()
        yaml.dump(sigopt_config, sys.stdout)
        n = timeout_input("Please confirm with sigopt parameters?(n for exit)", default = 'y')
        if n != 'y':
            exit()
        self.params['model_parameter'] = sigopt_config
        # 2. create sigopt connection
        self.conn, self.experiment = self.setup_sigopt_connection(sigopt_config, experiment_id)
        return self.experiment.id

    def setup_sigopt_connection(self, sigopt_config, experiment_id = None):
        name = sigopt_config["experiment"]
        parameters = sigopt_config["parameters"]
        metrics = sigopt_config["metrics"]
        observation_budget = sigopt_config["observation_budget"]
        project = sigopt_config["project"]

        num_tried = 0
        while True:
            try:
                conn = Connection()
                if experiment_id:
                    experiment = conn.experiments(experiment_id).fetch()
                else:
                    experiment = conn.experiments().create(
                        name=name,
                        parameters=parameters,
                        metrics=metrics,
                        observation_budget=observation_budget,
                        project=project)
                return conn, experiment
            except Exception as e:
                    num_tried += 1
                    print(f'[WARNING] Met exception when connecting to sigopt, will do retry in 5 secs, err msg is: ')
                    print(e)
                    if num_tried >= 30:
                        n = timeout_input("Retried connection for 30 times, do you still want to continue?(n for exit)", default = 'y', timeout = 10)
                        if n != 'y':
                            return None, None
                        num_tried = 0
                    time.sleep(5)

    def assignments_to_string(self, assignments):
        tmp = assignments
        if type(assignments) is not dict:
            tmp = json.loads(json.dumps(assignments))
        tmp = str(OrderedDict(sorted(tmp.items())))
        return tmp
 
    def assign_sigopt_suggestion(self, sigopt_assignments):
        config = {}
        for k, v in sigopt_assignments.items():
            config[k] = v  
        self.params['model_parameter']['tuned_parameters'] = config
        self.params['model_saved_path'] = os.path.join(self.params['save_path'], get_hash_string(str(config)))

    def record_assignmet(self, assignments, model_path):
        if not assignments:
            return
        tmp = self.assignments_to_string(assignments)
        self.assignment_model_tracker[tmp] = model_path

    def find_model_by_assignment(self, assignments):
        tmp = self.assignments_to_string(assignments)
        if tmp in self.assignment_model_tracker:
            return self.assignment_model_tracker[tmp]
        else:
            print(f"can't find {tmp} in {self.assignment_model_tracker}")
            return None

    def get_metrics_from_best_assignments(self, best_assignments):
        #Best Assignments: [BestAssignments(
        #  value=None,
        #  value_stddev=None,
        #  id="32401651",
        #  assignments=Assignments({
        #    "learning_rate": 0.9294458527831317,
        #    "max_depth": 11,
        #    "min_split_loss": 6.88375281543753
        #  }),
        #  values=[
        #    MetricEvaluation(
        #      value_stddev=None,
        #      value=0.7241269841269842,
        #      name="accuracy",
        #    ),
        #    MetricEvaluation(
        #      value_stddev=None,
        #      value=3.91267728805542,
        #      name="training_time",
        #    ),
        #  ],
        #)]
        metrics = []
        for item in best_assignments.values:
            metrics.append({"name": item.name, "value": item.value})
        return metrics

    def get_best_parameters_from_best_assignments(self, best_assignments):
        saved_path = os.path.join(self.params['save_path'], "best_parameters.yaml") 
        with open(saved_path, 'w') as f:
            yaml.dump(best_assignments.assignments, f)
        return best_assignments.assignments

    def launch_train_with_sigopt(self):
        budget = self.experiment.observation_budget
        self.logger.info(f'Experiment budget: {budget}')

        for _ in range(budget):
            suggestion = self.conn.experiments(self.experiment.id).suggestions().create()
            assignments = suggestion.assignments    
            self.assign_sigopt_suggestion(assignments)
            mkdir_or_backup_then_mkdir(self.params['model_saved_path'])
            model_path = ""
            metrics = []
            training_time, model_path, metrics = self.train_model(self.params)
            self.logger.info(f'Received sigopt suggestion, assignment is {self.assignments_to_string(assignments)}')
            self.record_assignmet(assignments, model_path)
            self.logger.info(f'Training completed with sigopt suggestion, metrics is {metrics}')
    
            num_tried = 0
            while True:
                try:
                    self.conn.experiments(self.experiment.id).observations().create(
                        suggestion=suggestion.id,
                        values=metrics,
                    )
                    self.experiment = self.conn.experiments(self.experiment.id).fetch()
                    break
                except Exception as e:
                    num_tried += 1
                    print(f'[WARNING] Met exception when connecting to sigopt, will do retry in 5 secs, err msg is:')
                    print(e)
                    if num_tried >= 30:
                        n = timeout_input("Retried connection for 30 times, do you still want to continue?(n for exit)", default = 'y', timeout = 10)
                        if n != 'y':
                            return model_path, metrics, assignments
                        num_tried = 0
                    time.sleep(5)
    
        self.logger.info(f"Training with sigopt is completed! https://app.sigopt.com/experiment/{self.experiment.id} ")
        all_best_assignments = self.conn.experiments(self.experiment.id).best_assignments().fetch()
        if len(all_best_assignments.data) == 0:
            self.logger.error("No assignments for satisfied model, you may increase observation budget or modify metric value")
            return model_path, metrics, assignments
        else:
            model_path = None
            best_assignments = self.get_best_parameters_from_best_assignments(all_best_assignments.data[0])
            model_path = self.find_model_by_assignment(best_assignments)
            metrics = self.get_metrics_from_best_assignments(all_best_assignments.data[0])
            return model_path, metrics, best_assignments

    def launch_train_without_sigopt(self):
        mkdir_or_backup_then_mkdir(self.params['model_saved_path'])
        training_time, model_path, metrics = self.train_model(self.params)
        self.logger.info(f'Training completed based in sigopt suggestion, took {training_time} secs')
        return model_path, metrics, self.params['model_parameter']["tuned_parameters"]
 
    def launch_train(self):
        if self.experiment != None:
            return self.launch_train_with_sigopt()
        else:
            return self.launch_train_without_sigopt()

    ###### Abstract method, Any advisor should have an implementation of below ######

    def generate_sigopt_yaml(self):
        # expect return sigopt config(dict)
        raise NotImplementedError("generate_sigopt_yaml is abstract.")

    def initialize_model_parameter(self, assignments = None):
        # expect return parameter(dict), this is for no_sigopt path
        raise NotImplementedError("initialize_model_parameter is abstract.")

    def train_model(self, args):
        # args has format as
        #  
        # expect to return training_time(float), model_path(str), metrics(list of dict)
        raise NotImplementedError("train_model is abstract.")

    def update_metrics(self):
        # expect return updated metrics(list of dict)
        raise NotImplementedError("update_metrics is abstract.")
