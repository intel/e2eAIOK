import ast
import os
import logging
import torch

from e2eAIOK.common.trainer.model.model_builder_nlp import ModelBuilderNLP
from e2eAIOK.DeNas.nlp.supernet_bert import SuperBertForQuestionAnswering, BertConfig
from e2eAIOK.DeNas.nlp.utils import decode_arch

class ModelBuilderNLPDeNas(ModelBuilderNLP):
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

    def _init_extra_model(self, model_path, model_structure):
        config = BertConfig.from_pretrained(model_path, num_labels=self.cfg.num_labels, finetuning_task=self.cfg.task_name)
        model = SuperBertForQuestionAnswering.from_pretrained(model_path, config)
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        model.to(device)
        if os.path.exists(model_structure):
            subbert_config = decode_arch(model_structure)
        else:
            subbert_config = {'sample_layer_num': 12, 'sample_num_attention_heads': [12] * 12, 'sample_qkv_sizes': [768] * 12, 'sample_hidden_size': 768, 'sample_intermediate_sizes': [3072] * 12}
        model.module.set_sample_config(subbert_config) if hasattr(model, 'module') else model.set_sample_config(subbert_config)
        size = model.module.calc_sampled_param_num() if hasattr(model, 'module') else model.calc_sampled_param_num()
        print("architecture: {}".format(subbert_config))
        print('Total parameters: {}'.format(size))
        return model