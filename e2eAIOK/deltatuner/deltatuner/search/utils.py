import sys
import os
import json
import time
import timeit
import yaml
import random
import torch
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig


class Timer:
    level = 0

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        print(f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def network_latency(model, tokenizer, batch_size=1, max_seq_length=32, infer_cnt=3.):
    batch, _ = input_constructor(batch_size, max_seq_length, tokenizer)
    aver_time = 0.
    model.eval()
    for i in range(int(infer_cnt)):
        start = time.time()
        with torch.no_grad():
            _ = model(**batch)
        end = time.time()
        sep = 1000 * (end - start)
        if i == 0:
            continue
        else:
            aver_time += sep / (infer_cnt)
    return aver_time

def input_constructor(batch_size, resolution, tokenizer):
    #tokenizer.pad_token = tokenizer.eos_token
    prompt = ["Once upon a time",
              "It was in the",
              "In his lifetime and",
              "the close links that",
              "In 2000 Boulter had",
              "a noted politician and",
              "He had an elder",
              "which he was attracted",
    ]
    prompt = prompt[:batch_size]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    return inputs, tokenizer

def network_is_legal(cand, vis_dict, params, supernet):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False
    cand = json.loads(cand)
    num_layers = sum(cand["num_hidden_layers"])
    if num_layers < 1:
        return False
    candidate_net = supernet.set_sample_config(cand)
    n_parameters = sum(param.numel() for param in candidate_net.parameters() if param.requires_grad)
    info['params'] = n_parameters / 10.**6
    if params.max_param_limits and info['params'] > params.max_param_limits:
        return False
    if params.min_param_limits and info['params'] < params.min_param_limits:
        return False
    info['visited'] = True
    return True

def populate_random_func(search_space, denas_config, supernet, search_space_name):
    cand_tuple = dict()
    for name in search_space_name:
        cand_tuple[name] = []
    for name in search_space_name:
        for i in range(getattr(supernet.config, denas_config.layer_name)):
            cand_tuple[name].append(random.choice(search_space[f"{name}_{i}"]))
    return json.dumps(cand_tuple)

def mutation_random_func(m_prob, s_prob, search_space, supernet, denas_config, top_candidates, search_space_name):
    cand = random.choice(top_candidates)
    result_cand = json.loads(cand)
    for i in range(getattr(supernet.config, denas_config.layer_name)):
        random_s = random.random()
        if random_s < s_prob:
            result_cand["num_hidden_layers"][i] = random.choice(search_space["num_hidden_layers_{}".format(i)])
    for name in search_space_name:
        if name == "num_hidden_layers":
            continue
        for i in range(getattr(supernet.config, denas_config.layer_name)):
            random_s = random.random()
            if random_s < m_prob:
                result_cand[name][i] = random.choice(search_space[f"{name}_{i}"])
    return json.dumps(result_cand)

def crossover_random_func(top_candidates, denas_config, supernet, search_space_name):
    p1 = json.loads(random.choice(top_candidates))
    p2 = json.loads(random.choice(top_candidates))
    max_iters_tmp = 50
    while p1 != p2 and max_iters_tmp > 0:
        max_iters_tmp -= 1
        p1 = json.loads(random.choice(top_candidates))
        p2 = json.loads(random.choice(top_candidates))
    cand = {}
    for name in search_space_name:
        cand[name] = []
    for name in search_space_name:
        for i in range(getattr(supernet.config, denas_config.layer_name)):        
            cand[name].append(random.choice([p1[name][i], p2[name][i]]))
    return json.dumps(cand)