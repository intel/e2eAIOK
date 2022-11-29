from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
from torch import nn
import torch.nn.functional as F

from e2eAIOK.DeNas.module.nlp.layernorm_super import LayerNormSuper as SuperBertLayerNorm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperEmbedding(nn.Module):
    def __init__(self, dict_size, embd_size, padding_idx=None):
        super(SuperEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_size, embd_size, padding_idx=padding_idx)
        self.sample_embedding_weight = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embedding_weight = self.embedding.weight[..., :sample_embed_dim]

    def calc_sampled_param_num(self):
        weight_numel = self.sample_embedding_weight.numel()
        assert weight_numel != 0
        return weight_numel

    def forward(self, input_ids):
        return F.embedding(input_ids, self.sample_embedding_weight.to(input_ids.device), self.embedding.padding_idx,
                           self.embedding.max_norm, self.embedding.norm_type,
                           self.embedding.scale_grad_by_freq, self.embedding.sparse)


class SuperBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(SuperBertEmbeddings, self).__init__()
        self.word_embeddings = SuperEmbedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SuperEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SuperEmbedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sample_embed_dim = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.word_embeddings.set_sample_config(sample_embed_dim)
        self.position_embeddings.set_sample_config(sample_embed_dim)
        self.token_type_embeddings.set_sample_config(sample_embed_dim)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def calc_sampled_param_num(self):
        w_emb_numel = self.word_embeddings.calc_sampled_param_num()
        p_emb_numel = self.position_embeddings.calc_sampled_param_num()
        t_emb_numel = self.token_type_embeddings.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        #logger.info('w_emb: {}\n'.format(w_emb_numel))
        #logger.info('p_emb: {}\n'.format(p_emb_numel))
        #logger.info('t_emb: {}\n'.format(t_emb_numel))
        #logger.info('ln_emb: {}\n'.format(ln_numel))

        return w_emb_numel + p_emb_numel + t_emb_numel + ln_numel

    def forward(self, input_ids, token_type_ids=None):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings += token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
