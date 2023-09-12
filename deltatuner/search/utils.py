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

def timeout_input(printout, default, timeout = None, interactive = True):
    if not interactive:
        return default
    import sys, select
    print(printout)
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if (i):
        msg = sys.stdin.readline().strip()
        return default if len(msg) == 0 else msg
    else:
        return default

def parse_config(conf_file):
    settings = {}
    if not os.path.exists(conf_file):
        return settings
    with open(conf_file) as f:
        settings.update(yaml.safe_load(f))
    return settings

def network_latency(model, tokenizer, batch_size=1, max_seq_length=32, infer_cnt=3.):
    batch, tok = input_constructor(batch_size, max_seq_length, tokenizer)
    aver_time = 0.
    model.eval()
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=max_seq_length,
        do_sample=True,
        pad_token_id = tok.eos_token_id
    )
    for i in range(int(infer_cnt)):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**batch, generation_config=generation_config)
        end = time.time()
        sep = 1000 * (end - start)
        if i == 0:
            continue
        else:
            aver_time += sep / (infer_cnt)
    return aver_time

def input_constructor(batch_size, resolution, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    prompt = ["Once upon a time,",
              "It was in the 11th century,",
              "In his lifetime and immediately",
              "However, as Hung notes,",
              "In 2000 Boulter had a guest",
              "a noted politician and poet",
              "He had an elder brother,",
              "which he was attracted after",
    ]
    prompt = prompt[:batch_size]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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