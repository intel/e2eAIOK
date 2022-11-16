import os
import time
import math
import logging
import torch
import trainer.extend_distributed as ext_dist

from tqdm import tqdm, trange
from module.nlp.optimization import BertAdam
from TorchTrainer import BaseTrainer
from data_builder import DataBuilder
from model.nlp.init_bert_parser import init_bert_parser
from model.nlp.bert_model_builder import BertModelBuilder
from model.nlp.utils import do_qa_eval, result_to_file

class BertTrainer(BaseTrainer):
    def __init__(self, args):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        parser = init_bert_parser()
        self.args = parser.parse_args(args)
        self.model_builder = BertModelBuilder(self.args)
        self.output_dir = self.args.output_dir
        self.task_name = "squad1"
        self.qa_tasks = ["squad1"]
        self.default_params = {
            "squad1": {"num_train_epochs": self.args.num_train_epochs, "max_seq_length": self.args.max_seq_length,
            "learning_rate": self.args.learning_rate, "eval_step": self.args.eval_step, "train_batch_size": self.args.train_batch_size},
        }
        ext_dist.init_distributed(backend=self.args.dist_backend)

    def train_one_epoch(self, model: torch.nn.Module, optimizer):
        
        model.train()
        tr_loss = 0.
        tr_cls_loss = 0.
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch

            if input_ids.size()[0] != self.args.train_batch_size:
                continue
            
            cls_loss = 0.
            cls_loss = model(input_ids, input_mask,
                            segment_ids, start_positions, end_positions)
            loss = cls_loss
            tr_cls_loss += cls_loss.item()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                self.global_step += 1

            if (self.global_step + 1) % self.args.eval_step == 0:
                res_f1 = self.evaluate(model, tr_loss, tr_cls_loss, step)
                if self.args.f1_threshold and res_f1 >= self.args.f1_threshold:
                    self.logger.info("Early stop at epoch {} iter {} with F1 {}".format(self.epoch_, self.global_step, res_f1))
                    self.is_stop = True
                    break

    def evaluate(self, model: torch.nn.Module, tr_loss, tr_cls_loss, step):
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Epoch = {} iter {} step".format(self.epoch_, self.global_step))
        self.logger.info("  Num examples = %d", len(self.eval_examples))

        model.eval()

        loss = tr_loss / (step + 1)
        cls_loss = tr_cls_loss / (step + 1)
        
        result = do_qa_eval(self.args, model, self.eval_dataloader, self.eval_features, self.eval_examples, self.device, self.eval_dataset, self.output_dir)
        self.infer_cnt += result['infer_cnt']
        self.infer_times.append(result['infer_time'])
        
        result['global_step'] = self.global_step
        result['cls_loss'] = cls_loss
        result['loss'] = loss
        result_to_file(result, self.output_eval_file)

        update_best = False
        if self.task_name in self.qa_tasks and result['f1'] + result['em'] > self.best_dev_acc:
            self.best_dev_acc = result['f1'] + result['em']
            self.best_dev_acc_str = 'f1: {}; em: {}'.format(result['f1'], result['em'])
            update_best = True

        if update_best:
            # Save a trained model
            model_name = "{}".format(self.task_name)
            logging.info("** ** * Saving fine-tuned model ** ** * ")
            # Only save the model it-self
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(self.output_dir, model_name)
            torch.save(model_to_save.state_dict(), output_model_file)
            self.tokenizer.save_vocabulary(self.output_dir)
        return result['f1']


    def fit(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if ext_dist.my_size > 1:
            self.default_params[self.task_name]['learning_rate'] =  self.default_params[self.task_name]['learning_rate'] * math.sqrt(int(ext_dist.my_size))
        
        if self.task_name in self.default_params:
            self.args.num_train_epochs = self.default_params[self.task_name]["num_train_epochs"]
            self.args.max_seq_length = self.default_params[self.task_name]["max_seq_length"]
            self.args.learning_rate = self.default_params[self.task_name]["learning_rate"]
            self.args.eval_step = self.default_params[self.task_name]["eval_step"]

            if 'train_batch_size' in self.default_params[self.task_name]:
                self.args.train_batch_size = self.default_params[self.task_name]["train_batch_size"]

        self.train_dataloader, self.eval_dataloader, self.train_examples, self.eval_examples, self.eval_dataset, self.eval_features, self.tokenizer = DataBuilder(self.args).get_data(ext_dist)
        model, self.subbert_config = self.model_builder.create_model(ext_dist)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if self.args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            self.args.gradient_accumulation_steps))

        self.args.train_batch_size = self.args.train_batch_size // self.args.gradient_accumulation_steps

        schedule = 'warmup_linear'
        num_train_optimization_steps = int(
            len(self.train_examples) / self.args.train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        self.logger.info("***** Running training *****")
        print("  Num examples = %d" % (len(self.train_examples)))
        print("  Batch size = %d" % (self.args.train_batch_size))
        print("  Num steps = %d" % (num_train_optimization_steps))

        # Train and evaluate
        self.is_stop = False
        self.global_step = 0
        self.best_dev_acc = 0.0
        self.best_dev_acc_str = ''
        self.infer_cnt = 0
        self.infer_times = []
        self.output_eval_file = os.path.join(self.output_dir, "eval_results.txt")

        start_time = time.time()
        for epoch_ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            if self.is_stop:
                break
            self.epoch_ = epoch_
            self.train_one_epoch(model, optimizer)
        end_time = time.time()
        model_to_save = model.module if hasattr(model, 'module') else model
        parameter_size = model_to_save.calc_sampled_param_num()
        output_str = "**************S*************\n" + \
                             "task_name = {}\n".format(self.task_name) + \
                             "architecture = {}\n".format(self.subbert_config) + \
                             "parameter size = {}\n".format(parameter_size) + \
                             "total training time = {}\n".format(str(end_time-start_time)) + \
                             "best_acc = %s\n" % self.best_dev_acc_str + \
                             "time_per_batch_infer = %.3f ms\n" % (sum(self.infer_times) / len(self.infer_times)) +\
                             "infer_cnt = %d\n" % self.infer_cnt +\
                             "**************E*************\n"
        print(output_str)
        output_eval_file = os.path.join(self.output_dir, "subbert.results")
        with open(output_eval_file, "a+") as writer:
            writer.write(output_str + '\n')