import os
import ast
import time
import random
import numpy as np
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from module.nlp.optimization import BertAdam
from nlp.supernet_bert import CrossEntropyQALoss
from nlp.utils_eval import do_qa_eval

from module.nlp.layernorm_super import LayerNormSuper
from module.nlp.Linear_super import LinearSuper
from thop.vision.basic_hooks import count_normalization, count_linear
from ptflops.pytorch_ops import linear_flops_counter_hook

def customer_ops_map_thop():
    customer_ops_map = {LayerNormSuper: count_normalization,
                        LinearSuper: count_linear}
    return customer_ops_map

def generate_search_space(search_space_config):
        # build arch space
        search_space = {}
        search_space['layer_num'] = range(int(search_space_config['LAYER_NUM']['bounds']['min']), int(search_space_config['LAYER_NUM']['bounds']['max'])+1)
        search_space['head_num'] = range(int(search_space_config['HEAD_NUM']['bounds']['min']), int(search_space_config['HEAD_NUM']['bounds']['max']) + int(search_space_config['HEAD_NUM']['bounds']['step']), int(search_space_config['HEAD_NUM']['bounds']['step']))
        search_space['hidden_size'] = range(int(search_space_config['HIDDEN_SIZE']['bounds']['min']), int(search_space_config['HIDDEN_SIZE']['bounds']['max']) + int(search_space_config['HIDDEN_SIZE']['bounds']['step']), int(search_space_config['HIDDEN_SIZE']['bounds']['step']))
        search_space['ffn_size'] = range(int(search_space_config['INTERMEDIATE_SIZE']['bounds']['min']), int(search_space_config['INTERMEDIATE_SIZE']['bounds']['max']) + int(search_space_config['INTERMEDIATE_SIZE']['bounds']['step']), int(search_space_config['INTERMEDIATE_SIZE']['bounds']['step']))
        return search_space

def get_subconfig(cand):
    subconfig = dict()
    subconfig['sample_layer_num'] = cand[0]
    subconfig['sample_num_attention_heads'] = [cand[1]] * cand[0]
    subconfig['sample_qkv_sizes'] = [cand[2]] * cand[0]
    subconfig['sample_hidden_size'] = cand[3]
    subconfig['sample_intermediate_sizes'] = [cand[4]] * cand[0]
    return subconfig

def decode_arch(arches_file):
    subbert_config = None
    with open(arches_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            subbert_config = get_subconfig(ast.literal_eval(line))
    return subbert_config

def bert_populate_random_func(search_space):
    cand_tuple = list() #[layer_num, [num_attention_heads]*layer_num, [qkv_sizes]*layer_num, hidden_size, [intermediate_sizes]*layer_num]
    dimensions = ['head_num', 'hidden_size', 'ffn_size']
    depth = random.choice(search_space['layer_num'])
    cand_tuple.append(depth)
    for dimension in dimensions:
        if dimension == 'head_num':
            head_num = random.choice(search_space[dimension])
            qkv_size = head_num * 64
            cand_tuple.append(head_num)
            cand_tuple.append(qkv_size)
        elif dimension == 'hidden_size':
            cand_tuple.append(random.choice(search_space['hidden_size']))
        elif dimension == 'ffn_size':
            cand_tuple.append(random.choice(search_space['ffn_size']))

    return tuple(cand_tuple)

def bert_is_legal(cand, vis_dict, params, super_net):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False
    subconfig = get_subconfig(cand)
    super_net.set_sample_config(subconfig)
    n_parameters = super_net.calc_sampled_param_num()
    info['params'] = n_parameters / 10.**6
    if info['params'] > params.max_param_limits:
        return False
    if info['params'] < params.min_param_limits:
        return False
    info['visited'] = True
    return True

def bert_mutation_random_func(m_prob, s_prob, search_space, top_candidates):
    cand = list(random.choice(top_candidates))
    depth, num_heads, qkv_sizes, hidden_size, ffn_sizes = cand[0], cand[1], cand[2], cand[3], cand[4]
    random_s = random.random()
    # depth
    if random_s < s_prob:
        new_depth = random.choice(search_space['layer_num'])
        depth = new_depth
        num_heads = random.choice(search_space['head_num'])
        qkv_sizes = num_heads * 64
        hidden_size = random.choice(search_space['hidden_size'])
        ffn_sizes = random.choice(search_space['ffn_size'])
    random_s = random.random()
    if random_s < m_prob:
        # num_heads
        num_heads = random.choice(search_space['head_num'])
        # qkv_sizes
        qkv_sizes = num_heads * 64
    # hidden_size
    random_s = random.random()
    if random_s < s_prob:
        hidden_size = random.choice(search_space['hidden_size'])
    # ffn_sizes
    random_s = random.random()
    if random_s < s_prob:
        ffn_sizes = random.choice(search_space['ffn_size'])

    result_cand = [depth] + [num_heads] + [qkv_sizes] + [hidden_size] + [ffn_sizes]
    return tuple(result_cand)

def bert_crossover_random_func(top_candidates):
    p1 = random.choice(top_candidates)
    p2 = random.choice(top_candidates)
    max_iters_tmp = 50
    while len(p1) != len(p2) and max_iters_tmp > 0:
        max_iters_tmp -= 1
        p1 = random.choice(top_candidates)
        p2 = random.choice(top_candidates)
    cand = []
    for ind, it in enumerate(zip(p1, p2)):
        if ind == 2:
            continue
        elif ind == 1:
            cand.append(random.choice(it))
            cand.append(cand[-1] * 64)
        else:
            cand.append(random.choice(it))
    return tuple(cand)

def get_bert_latency(model, batch_size, max_seq_length, gpu, infer_cnt):
    if gpu is None:
        device = 'cpu'
    else:
        device = 'cuda'
    input_ids = [9333] * max_seq_length
    input_masks = max_seq_length * [1]
    input_segments = max_seq_length * [0]
    input_ids = torch.tensor([input_ids]*batch_size, dtype=torch.long).to(device)
    input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long).to(device)
    input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long).to(device)

    aver_time = 0.
    model.eval()

    for i in range(int(infer_cnt)):
        start = time.time()
        with torch.no_grad():
            model.forward(input_ids, input_masks, input_segments)

        end = time.time()
        sep = 1000 * (end - start)

        if i == 0:
            continue
        else:
            aver_time += sep / (infer_cnt - 1)
    return aver_time

def bert_create_optimizer(model, cfg):
    if cfg.optimizer == "BertAdam":
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = int(
            cfg.num_train_steps / cfg.gradient_accumulation_steps) * cfg.train_epochs
        if ext_dist.my_size > 1:
            num_train_optimization_steps = num_train_optimization_steps // ext_dist.my_size
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=cfg.lr_scheduler,
                             lr=cfg.learning_rate,
                             warmup=cfg.warmup_proportion,
                             t_total=num_train_optimization_steps,
                             weight_decay=cfg.weight_decay)
        return optimizer

def bert_create_criterion(cfg):
    if cfg.criterion == "CrossEntropyQALoss":
        criterion = CrossEntropyQALoss(cfg.max_seq_length)
    return criterion

def bert_create_scheduler(cfg):
    return None

def bert_create_metric(cfg):
    if cfg.eval_metric == "qa_f1":
        return do_qa_eval