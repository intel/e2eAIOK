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
from torch.nn.modules import linear
from transformers import pytorch_utils

from e2eAIOK.DeNas.thirdparty.supernet_hf import SuperHFModel

LINEAR_LAYER_STRUCTURE = {"output": linear.Linear,
                            "intermediate": linear.Linear,
                            "mlp": pytorch_utils.Conv1D
                        }

ATTN_LAYER_STRUCTURE = {"selfatt": linear.Linear,
                            "att": pytorch_utils.Conv1D
                        }

def decode_arch(arches_file):
    subhf_config = None
    with open(arches_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            subhf_config = ast.literal_eval(line)
    return subhf_config

def hf_populate_random_func(search_space):
    cand_tuple = dict()
    for search_key in search_space:
        cand_tuple[search_key] = random.choice(search_space[search_key]) #nosec

    cand_tuple["hidden_size"] = int(cand_tuple["hidden_size"]/cand_tuple["num_attention_heads"]) * cand_tuple["num_attention_heads"]

    return json.dumps(cand_tuple)

def hf_is_legal(cand, vis_dict, params):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False
    cand = json.loads(cand)
    candidate_net = SuperHFModel.set_sample_config(params.supernet, **cand)
    n_parameters = sum(param.numel() for param in candidate_net.parameters())
    info['params'] = n_parameters / 10.**6
    if info['params'] > params.max_param_limits:
        return False
    if info['params'] < params.min_param_limits:
        return False
    info['visited'] = True
    return True

def hf_mutation_random_func(m_prob, s_prob, search_space, top_candidates):
    cand = random.choice(top_candidates) #nosec
    random_s = random.random() #nosec
    result_cand = json.loads(cand)
    
    if random_s < m_prob:
        # num_heads
        result_cand['num_attention_heads'] = random.choice(search_space['num_attention_heads']) #nosec
    for search_key in search_space:
        if search_key == 'num_attention_heads':
            continue
        random_s = random.random() #nosec
        if random_s < s_prob:
            result_cand[search_key] = random.choice(search_space[search_key]) #nosec

    result_cand["hidden_size"] = int(result_cand["hidden_size"]/result_cand["num_attention_heads"]) * result_cand["num_attention_heads"]

    return json.dumps(result_cand)

def hf_crossover_random_func(top_candidates):
    p1 = json.loads(random.choice(top_candidates)) #nosec
    p2 = json.loads(random.choice(top_candidates)) #nosec
    max_iters_tmp = 50
    while len(p1) != len(p2) and max_iters_tmp > 0:
        max_iters_tmp -= 1
        p1 = json.loads(random.choice(top_candidates)) #nosec
        p2 = json.loads(random.choice(top_candidates)) #nosec
    cand = {}
    for it1, it2 in zip(p1, p2):
        select_item = random.choice([p1[it1], p2[it2]]) #nosec
        cand[it1] = select_item
    cand["hidden_size"] = int(cand["hidden_size"]/cand["num_attention_heads"]) * cand["num_attention_heads"]
    return json.dumps(cand)

def get_hf_latency(model, batch_size, max_seq_length, gpu, infer_cnt):
    if gpu is None:
        device = 'cpu'
    else:
        device = 'cuda'
    
    batch = input_construtor(batch_size, max_seq_length)

    aver_time = 0.
    model.eval()

    for i in range(int(infer_cnt)):
        start = time.time()
        with torch.no_grad():
            model(**batch)
        end = time.time()
        sep = 1000 * (end - start)
        if i == 0:
            continue
        else:
            aver_time += sep / (infer_cnt - 1)
    return aver_time

def input_construtor(batch_size, resolution, domain="NLP"):
    if domain == "NLP":
        input_ids = [[9333-id] * resolution for id in range(batch_size)]
        input_masks = resolution * [1]
        input_segments = resolution * [0]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long)
        input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long)
        batch = {'input_ids': input_ids, 'token_type_ids': input_segments, 'attention_mask':input_masks}
        return batch
    else:
        raise NotImplementedError