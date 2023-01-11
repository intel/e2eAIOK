import subprocess
import yaml
import logging
import os

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.modeladvisor.BaseModelAdvisor import BaseModelAdvisor

class BERTAdvisor(BaseModelAdvisor):
    '''
        Bert sigopt model optimization launcher
    '''
    def __init__(self, dataset_meta_path, train_path, eval_path, args):
        super().__init__(dataset_meta_path, train_path, eval_path, args)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')
        self.python_path = "/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/"
        self.train_python = f"{self.python_path}/python"
        self.train_script = os.path.join(os.getcwd(),"modelzoo/bert/benchmarks/launch_benchmark.py")
        self.train_path = train_path
        self.test_path = eval_path
        self.dataset_meta_path = dataset_meta_path
        self.current_path = os.getcwd()
        self.extra_pythonpath = f"{self.current_path}/modelzoo/bert/benchmarks/"

    def initialize_model_parameter(self, assignments=None):
        '''
            Add model specific parameters
            For sigopt parameters, set parameter explicitly and the parameter will not be optimized by sigopt
        '''
        config = {}
        tuned_parameters = {}
        if assignments:
            tuned_parameters["train_batch_size"] = str(assignments['train_batch_size'])
            tuned_parameters["learning_rate"] = str(assignments['learning_rate'])
            tuned_parameters["warmup_proportion"] = str(assignments['warmup_proportion'])
            tuned_parameters["num_hidden_layers"] = str(assignments['num_hidden_layers'])
            if 'dropout_prob' in assignments:
                tuned_parameters["attention_probs_dropout_prob"] = str(assignments['dropout_prob'])
                tuned_parameters["hidden_dropout_prob"] = str(assignments['dropout_prob'])
            else:
                tuned_parameters["attention_probs_dropout_prob"] = str(assignments['attention_probs_dropout_prob'])
                tuned_parameters["hidden_dropout_prob"] = str(assignments['hidden_dropout_prob'])
        else:
            tuned_parameters["train_batch_size"] = "24"
            tuned_parameters["learning_rate"] = "3e-5"
            tuned_parameters["warmup_proportion"] = "0.1"
            tuned_parameters["num_hidden_layers"] = "24"
            tuned_parameters["attention_probs_dropout_prob"] = "0.1"
            tuned_parameters["hidden_dropout_prob"] = "0.1"
        config['tuned_parameters'] = tuned_parameters
        self.params['model_parameter'] = config
        self.params['model_saved_path'] = os.path.join(
            self.params['save_path'], get_hash_string(str(tuned_parameters)))
        return config

    ###### Implementation of required methods ######
    
    def generate_sigopt_yaml(self, file='bert_sigopt.yaml'):
        config = {}
        config['project'] = 'bert'
        config['experiment'] = 'BERT'
        parameters = [
        {'name':'learning_rate','bounds':{'min':1e-6, 'max':3e-4},'type':'double'},
        {'name':'train_batch_size','grid':[16,24,32,64,96],'type':'int'},
        {'name':'num_hidden_layers', 'grid':[8,12,16,20,24], 'type':'int'},
        {'name':'dropout_prob', 'grid':[0.08,0.1,0.12], 'type':'double'},
        {'name':'warmup_proportion', 'bounds':{'min':0.05, 'max':0.20},'type':'double'},
        ]
        config['parameters'] = parameters
        metrics = []
        if 'training_time_threshold' in self.params:
            metrics.append({'name': 'training_time', 'objective': 'minimize', 'threshold': self.params['training_time_threshold']})
        else:
            metrics.append({'name': 'training_time', 'objective': 'minimize'})
        if 'metric_threshold' in self.params:
            metrics.append({'name': self.params['metric'], 'objective': self.params['metric_objective'], 'threshold': self.params['metric_threshold']})
        else:
            metrics.append({'name': self.params['metric'], 'objective': self.params['metric_objective']})
        
        config['metrics'] = metrics
        config['observation_budget'] = self.params['observation_budget']

        with open(file, 'w') as f:
            yaml.dump(config, f)
        return config

    def update_metrics(self):
        metrics = []
        metrics.append({'name': 'f1', 'value': self.mean_accuracy})
        metrics.append({'name': 'training_time', 'value': self.training_time})
        self.params['model_parameter']['metrics'] = metrics
        return self.params['model_parameter']['metrics']

    def train_model(self, args):
        model_path = args['model_saved_path']
        if int(args['mpi_num_processes']) > 1:
            self.dist_launch(args)
        else:
            self.launch(args)
        with open(os.path.join(self.params['model_saved_path'],"best_auc.txt"),'r') as f:
            lines = f.readlines()
        self.mean_accuracy = float(lines[-1])
        with open(os.path.join(self.params['model_saved_path'],"best_time.txt"),'r') as f:
            lines = f.readlines()
        self.training_time = float(lines[-1])
        metrics = self.update_metrics()
        return self.training_time, model_path, metrics

    def dist_launch(self, args):
        # construct BERT launch command
        os.environ["MODEL_DIR"] = MODEL_DIR = os.path.join(os.getcwd(),"modelzoo/bert")
        os.environ["OUTPUT_DIR"] = OUT_DIR = self.params['model_saved_path']
        os.environ["DATASET_DIR"] = os.path.dirname(self.dataset_meta_path)
        os.environ["CHECKPOINT_DIR"] = "$DATASET_DIR/pre-trained-model/bert-large-uncased/wwm_uncased_L-24_H-1024_A-16"
        os.environ["HOROVOD_CPU_OPERATIONS"] = "CCL"
        os.environ["NOINSTALL"] = "True"
        os.environ["CCL_WORKER_COUNT"] = str(args['ccl_worker_num'])
        os.environ["CCL_WORKER_AFFINITY"] = "16,17,34,35"
        os.environ["HOROVOD_THREAD_AFFINITY"] = "53,71"
        os.environ["I_MPI_PIN_DOMAIN"] = "socket"
        os.environ["I_MPI_PIN_PROCESSOR_EXCLUDE_LIST"] = "16,17,34,35,52,53,70,71"
        mpi_num_proc_arg = ""

        if not os.path.exists(os.environ["OUTPUT_DIR"]):
            os.makedirs(os.environ["OUTPUT_DIR"])

        cmd = f"PYTHONPATH=$PYTHONPATH:{self.extra_pythonpath} "
        cmd += f"{self.train_python} "
        cmd += f"{MODEL_DIR}/benchmarks/launch_benchmark.py " \
        + f"--model-name=bert_large " \
        + f"--precision=fp32 " \
        + f"--mode=training " \
        + f"--framework=tensorflow {mpi_num_proc_arg}" \
        + f"--batch-size={args['model_parameter']['tuned_parameters']['train_batch_size']} " \
        + f"--output-dir {OUT_DIR} " \
        + f"--host_file=$MODEL_DIR/hosts " \
        + f"--mpi_num_processes {args['mpi_num_processes']} " \
        + f"--num-intra-threads 36 " \
        + f"--num-inter-threads 2 " \
        + f"--train_option=SQuAD " \
        + f"vocab_file=$CHECKPOINT_DIR/vocab.txt " \
        + f"config_file=$CHECKPOINT_DIR/bert_config.json " \
        + f"init_checkpoint=$CHECKPOINT_DIR/bert_model.ckpt " \
        + f"do_train=True " \
        + f"train_file=$DATASET_DIR/train-v1.1.json " \
        + f"do_predict=True " \
        + f"predict_file=$DATASET_DIR/dev-v1.1.json " \
        + f"test_file=$DATASET_DIR/test-v1.1.json " \
        + f"data_dir=$DATASET_DIR " \
        + f"num_to_evaluate=50 " \
        + f"step_threshold={args['step_threshold']} " \
        + f"f1_threshold=90.87 " \
        + f"num_train_epochs=2 " \
        + f"max_seq_length=384 " \
        + f"doc_stride=128 " \
        + f"optimized_softmax=True " \
        + f"experimental_gelu=False " \
        + f"do_lower_case=True " \
        + f"num_hidden_layers={args['model_parameter']['tuned_parameters']['num_hidden_layers']} " \
        + f"learning_rate={args['model_parameter']['tuned_parameters']['learning_rate']} " \
        + f"attention_probs_dropout_prob={args['model_parameter']['tuned_parameters']['attention_probs_dropout_prob']} " \
        + f"hidden_dropout_prob={args['model_parameter']['tuned_parameters']['hidden_dropout_prob']} " \
        + f"warmup_proportion={args['model_parameter']['tuned_parameters']['warmup_proportion']} "

        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def launch(self, args):
        # construct BERT launch command
        os.environ["MODEL_DIR"] = MODEL_DIR = os.path.join(os.getcwd(),"modelzoo/bert")
        os.environ["OUTPUT_DIR"] = OUT_DIR = self.params['model_saved_path']
        os.environ["DATASET_DIR"] = os.path.dirname(self.dataset_meta_path)
        os.environ["CHECKPOINT_DIR"] = "$DATASET_DIR/pre-trained-model/bert-large-uncased/wwm_uncased_L-24_H-1024_A-16"
        mpi_num_proc_arg = ""

        if not os.path.exists(os.environ["OUTPUT_DIR"]):
            os.makedirs(os.environ["OUTPUT_DIR"])

        cmd = f"PYTHONPATH=$PYTHONPATH:{self.extra_pythonpath} "
        cmd += f"{self.train_python} "
        cmd += f"{MODEL_DIR}/benchmarks/launch_benchmark.py " \
        + f"--model-name=bert_large " \
        + f"--precision=fp32 " \
        + f"--mode=training " \
        + f"--framework=tensorflow {mpi_num_proc_arg}" \
        + f"--batch-size={args['model_parameter']['tuned_parameters']['train_batch_size']} " \
        + f"--output-dir {OUT_DIR} " \
        + f"--host_file=$MODEL_DIR/hosts " \
        + f"--num-intra-threads 36 " \
        + f"--num-inter-threads 2 " \
        + f"--train_option=SQuAD " \
        + f"vocab_file=$CHECKPOINT_DIR/vocab.txt " \
        + f"config_file=$CHECKPOINT_DIR/bert_config.json " \
        + f"init_checkpoint=$CHECKPOINT_DIR/bert_model.ckpt " \
        + f"do_train=True " \
        + f"train_file=$DATASET_DIR/train-v1.1.json " \
        + f"do_predict=True " \
        + f"predict_file=$DATASET_DIR/dev-v1.1.json " \
        + f"test_file=$DATASET_DIR/test-v1.1.json " \
        + f"data_dir=$DATASET_DIR " \
        + f"num_to_evaluate=50 " \
        + f"step_threshold={args['step_threshold']} " \
        + f"f1_threshold=90.87 " \
        + f"num_train_epochs=2 " \
        + f"max_seq_length=384 " \
        + f"doc_stride=128 " \
        + f"optimized_softmax=True " \
        + f"experimental_gelu=False " \
        + f"do_lower_case=True " \
        + f"num_hidden_layers={args['model_parameter']['tuned_parameters']['num_hidden_layers']} " \
        + f"learning_rate={args['model_parameter']['tuned_parameters']['learning_rate']} " \
        + f"attention_probs_dropout_prob={args['model_parameter']['tuned_parameters']['attention_probs_dropout_prob']} " \
        + f"hidden_dropout_prob={args['model_parameter']['tuned_parameters']['hidden_dropout_prob']} " \
        + f"warmup_proportion={args['model_parameter']['tuned_parameters']['warmup_proportion']} "

        print(f'training launch command: {cmd}')
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
