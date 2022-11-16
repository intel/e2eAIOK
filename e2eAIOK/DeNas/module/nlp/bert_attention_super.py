# coding=utf-8
# Copyright (c) 2022, Intel. and its affiliates.
# Copyright 2021 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
from torch import nn

from module.nlp.Linear_super import LinearSuper as SuperLinear
from module.nlp.layernorm_super import LayerNormSuper as SuperBertLayerNorm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfAttention, self).__init__()
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(qkv_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = SuperLinear(config.hidden_size, self.all_head_size)
        self.key = SuperLinear(config.hidden_size, self.all_head_size)
        self.value = SuperLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.sample_num_attention_head = None
        self.sample_attention_head_size = None
        self.sample_qkv_size = None

    def set_sample_config(self, sample_embed_dim, num_attention_head, qkv_size,
                          in_index=None, out_index=None):
        assert qkv_size % num_attention_head == 0
        self.sample_qkv_size = qkv_size
        self.sample_attention_head_size = qkv_size // num_attention_head
        self.sample_num_attention_head = num_attention_head

        self.query.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)
        self.key.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)
        self.value.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)

    def calc_sampled_param_num(self):
        query_numel = self.query.calc_sampled_param_num()
        key_numel = self.key.calc_sampled_param_num()
        value_numel = self.value.calc_sampled_param_num()

        #logger.info('query_numel: {}\n'.format(query_numel))
        #logger.info('key_numel: {}\n'.format(key_numel))
        #logger.info('value_numel: {}\n'.format(value_numel))

        return query_numel + key_numel + value_numel

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.sample_num_attention_head, self.sample_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.sample_attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(self.dropout(attention_probs), value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.sample_qkv_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class SuperBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfOutput, self).__init__()
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size

        self.dense = SuperLinear(qkv_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, qkv_size, sample_embed_dim, in_index=None):
        self.dense.set_sample_config(qkv_size, sample_embed_dim, in_index=in_index)
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


class SuperBertAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertAttention, self).__init__()
        self.self = SuperBertSelfAttention(config)
        self.output = SuperBertSelfOutput(config)

    def set_sample_config(self, sample_embed_dim, num_attention_head, qkv_size):
        self.self.set_sample_config(sample_embed_dim, num_attention_head, qkv_size)
        self.output.set_sample_config(qkv_size, sample_embed_dim)

    def calc_sampled_param_num(self):
        self_numel = self.self.calc_sampled_param_num()
        output_numel = self.output.calc_sampled_param_num()

        #logger.info('self_numel: {}\n'.format(self_numel))
        #logger.info('output_numel: {}\n'.format(output_numel))

        return self_numel + output_numel

    def forward(self, input_tensor, attention_mask):

        self_output = self.self(input_tensor, attention_mask)
        self_output, layer_att = self_output
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att
