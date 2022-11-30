from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import copy

import torch
from torch import nn

from module.nlp.Linear_super import LinearSuper as SuperLinear
from module.nlp.layernorm_super import LayerNormSuper as SuperBertLayerNorm
from module.nlp.bert_attention_super import SuperBertAttention
from module.nlp.bert_intermediate_super import SuperBertIntermediate


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperBertOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertOutput, self).__init__()
        self.dense = SuperLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, intermediate_size, sample_embed_dim):
        self.dense.set_sample_config(intermediate_size, sample_embed_dim)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        #logger.info('dense_numel: {}\n'.format(dense_numel))
        #logger.info('ln_numel: {}\n'.format(ln_numel))
        return dense_numel + ln_numel

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperBertLayer(nn.Module):
    def __init__(self, config):
        super(SuperBertLayer, self).__init__()
        self.attention = SuperBertAttention(config)
        self.intermediate = SuperBertIntermediate(config)
        self.output = SuperBertOutput(config)

    def set_sample_config(self, sample_embed_dim, intermediate_size, num_attention_head, qkv_size):
        self.attention.set_sample_config(sample_embed_dim, num_attention_head, qkv_size)
        self.intermediate.set_sample_config(sample_embed_dim, intermediate_size)
        self.output.set_sample_config(intermediate_size, sample_embed_dim)

    def calc_sampled_param_num(self):
        attention_numel = self.attention.calc_sampled_param_num()
        intermediate_numel = self.intermediate.calc_sampled_param_num()
        output_numel = self.output.calc_sampled_param_num()

        #logger.info('attention_numel: {}\n'.format(attention_numel))
        #logger.info('intermediate_numel: {}\n'.format(intermediate_numel))
        #logger.info('output_numel: {}\n'.format(output_numel))

        return attention_numel + intermediate_numel + output_numel

    def forward(self, hidden_states, attention_mask):

        attention_output = self.attention(hidden_states, attention_mask)
        attention_output, layer_att = attention_output

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, layer_att


class SuperBertEncoder(nn.Module):
    def __init__(self, config):
        super(SuperBertEncoder, self).__init__()
        layer = SuperBertLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.sample_layer_num = None

    def set_sample_config(self, subbert_config):
        self.sample_layer_num = subbert_config['sample_layer_num']
        self.sample_embed_dim = subbert_config['sample_hidden_size']
        self.num_attention_heads = subbert_config['sample_num_attention_heads']
        self.itermediate_sizes = subbert_config['sample_intermediate_sizes']
        self.qkv_sizes = subbert_config['sample_qkv_sizes']
        for layer, num_attention_head, intermediate_size, qkv_size in zip(self.layers[:self.sample_layer_num],
                                                                          self.num_attention_heads,
                                                                          self.itermediate_sizes,
                                                                          self.qkv_sizes):
            layer.set_sample_config(self.sample_embed_dim, intermediate_size, num_attention_head, qkv_size)

    def calc_sampled_param_num(self):
        layers_numel = 0

        for layer in self.layers[:self.sample_layer_num]:
            layers_numel += layer.calc_sampled_param_num()

        #logger.info('layer_numel: {}'.format(layers_numel))

        return layers_numel

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_att = []

        for i, layer_module in enumerate(self.layers[:self.sample_layer_num]):
            all_encoder_layers.append(hidden_states)
            hidden_states = layer_module(all_encoder_layers[i], attention_mask)
            hidden_states, layer_att = hidden_states
            all_encoder_att.append(layer_att)

        all_encoder_layers.append(hidden_states)

        return all_encoder_layers, all_encoder_att

