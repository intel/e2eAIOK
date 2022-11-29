from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math
import logging

import torch
from torch import nn

from e2eAIOK.DeNas.module.nlp.Linear_super import LinearSuper as SuperLinear


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class SuperBertIntermediate(nn.Module):
    def __init__(self, config):
        super(SuperBertIntermediate, self).__init__()
        self.dense = SuperLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def set_sample_config(self, sample_embed_dim, intermediate_size):
        self.dense.set_sample_config(sample_embed_dim, intermediate_size)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()

        #logger.info('dense_numel: {}\n'.format(dense_numel))
        return dense_numel

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
