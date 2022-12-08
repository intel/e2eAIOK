import ast
import os
import logging
import torch

from e2eAIOK.common.trainer.model_builder import ModelBuilder
from e2eAIOK.DeNas.nlp.supernet_bert import SuperBertForQuestionAnswering, BertConfig
from e2eAIOK.DeNas.nlp.utils import decode_arch

class ModelBuilderNLP(ModelBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_model(self):
        config = BertConfig.from_pretrained(self.cfg.model, num_labels=self.cfg.num_labels, finetuning_task=self.cfg.task_name)
        model = SuperBertForQuestionAnswering.from_pretrained(self.cfg.model, config)
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        model.to(device)
        subbert_config = decode_arch(self.cfg.best_model_structure)
        model.module.set_sample_config(subbert_config) if hasattr(model, 'module') else model.set_sample_config(subbert_config)
        size = model.module.calc_sampled_param_num() if hasattr(model, 'module') else model.calc_sampled_param_num()
        print("architecture: {}".format(subbert_config))
        print('Total parameters: {}'.format(size))
        return model