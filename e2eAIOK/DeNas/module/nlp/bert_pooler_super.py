from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn

from module.nlp.Linear_super import LinearSuper as SuperLinear


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperBertPooler(nn.Module):
    def __init__(self, config):
        super(SuperBertPooler, self).__init__()
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def set_sample_config(self, sample_hidden_dim):
        self.dense.set_sample_config(sample_hidden_dim, sample_hidden_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        #logger.info('dense_numel: {}'.format(dense_numel))

        return dense_numel

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
