import subprocess
import yaml
import logging
import time
import os
import pandas as pd

from AIDK.common.utils import *
from AIDK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class TPCxAIAdvisor09(BaseModelAdvisor):
    def __init__(self, dataset_meta_path, train_path, eval_path, settings):
        super().__init__(dataset_meta_path, train_path, eval_path, settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')
        self.train_path = train_path
        self.test_path = eval_path
        self.dataset_meta_path = dataset_meta_path
        self.saved_path = self.params['save_path']
        self.data_path = settings['data_path']
        self.uc9res_path = f"{settings['data_path']}/uc9_res"

    def update_metrics(self):
        cmd = f"spark-submit --files {self.uc9res_path}/shape_predictor_5_face_landmarks.dat,{self.uc9res_path}/nn4.small2.v1.h5 /home/vmagent/app/hydro.ai/modelzoo/TPCxAI/tpcxai_uc09/UseCase09.py --stage serving --workdir output/model/uc09 --output output/model/uc09/SCORING_1 'output/data/valid/CUSTOMER_IMAGES_META.csv' 'output/data/valid/CUSTOMER_IMAGES.seq'"
        self.logger.info(f'serving launch command: {cmd}')
        os.chdir('/tmp/')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        predictions_file = "/home/vmagent/app/hydro.ai/modelzoo/TPCxAI/tpcxai_uc09/predictions.csv"
        cmd_pred = f"hdfs dfs -cat output/model/uc09/SCORING_1/predictions.csv/* | awk " \
             + "'BEGIN{f=" \
             + '""' \
             + "}{if($0!=f){print $0}if(NR==1){f=$0}}' > " \
             + f"{predictions_file}"
        process_pred = subprocess.Popen(cmd_pred, shell=True)
        process_pred.wait()
        df_truth = pd.read_csv(self.data_path+"/valid/CUSTOMER_IMAGES_META_labels.csv")
        my_dict = {row[2]: row[0] for row in df_truth.values}
        df_pred = pd.read_csv(predictions_file)
        num_truth=0
        num_total=0
        for index, row in df_pred.iterrows():
            num_total=num_total+1
            if(row[1]==my_dict.get(row[0])):
                num_truth=num_truth+1
        self.mean_accuracy = num_truth/num_total
        metrics = []
        metrics.append({'name': 'accuracy', 'value': self.mean_accuracy})
        metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def initialize_model_parameter(self, assignments=None):
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters['epochs_embedding'] = assignments['epochs_embedding']
            tuned_parameters['batch'] = assignments['batch']
        else:
            tuned_parameters['epochs_embedding'] = 15
            tuned_parameters['batch'] = 64
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    def generate_sigopt_yaml(self, file='tpcxaiuc9_sigopt.yaml'):
        config = {}
        config['project'] = self.params['model_parameter']['project']
        config['experiment'] = self.params['model_parameter']['experiment']
        parameters = [
        {
            'name': 'epochs_embedding',
            'bounds': {
                'min': 10,
                'max': 20 
            },
            'type': 'int',
        }, {
            'name': 'batch',
            'bounds': {
                'min': 32,
                'max': 70
            },
            'type': 'int'
        }]
        user_defined_parameter = self.params['model_parameter']['parameters'] if ('model_parameter' in self.params) and ('parameters' in self.params['model_parameter']) else None
        config['parameters'] = parameters
        if user_defined_parameter:
            self.logger.info(
                f"Update with user defined parameters {user_defined_parameter}"
            )
            update_list(config['parameters'], user_defined_parameter)
        config['metrics'] = [{
            'name': 'accuracy',
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

        saved_path = os.path.join(self.params['save_path'], file)
        with open(saved_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def train_model(self, args):
        start_time = time.time()
        self.launch(args)
        self.training_time = time.time() - start_time
        self.logger.info(f'calculated training time: {self.training_time}')
        metrics = self.update_metrics()
        model_path = args['model_saved_path']
        return self.training_time, model_path, metrics

    def launch(self, args):
        cmd = f"spark-submit --files {self.uc9res_path}/shape_predictor_5_face_landmarks.dat,{self.uc9res_path}/nn4.small2.v1.h5 /home/vmagent/app/hydro.ai/modelzoo/TPCxAI/tpcxai_uc09/UseCase09.py "
        cmd += f"--stage=training " \
            + f"--epochs_embedding={args['model_parameter']['tuned_parameters']['epochs_embedding']} " \
            + f"--batch={args['model_parameter']['tuned_parameters']['batch']} " \
            + f"--executor_cores_horovod 1 " \
            + f"--task_cpus_horovod 1 " \
            + f"--workdir output/model/uc09 " \
            + f"'output/data/train/CUSTOMER_IMAGES_META.csv' 'output/data/train/CUSTOMER_IMAGES.seq'"
        self.logger.info(f'training launch command: {cmd}')
        os.chdir('/tmp/')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
